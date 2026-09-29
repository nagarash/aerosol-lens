"""Hourly MERRA-2 storage and six-hour binary frame batches.

Each immutable variable/day Zarr is published by atomic rename after validation.
Daily means are never used. Request-time misses fetch only selected cells/hours.
Full-day ingestion is an explicit CLI operation; archive jobs are serialized.
Run: python -m backend.hourly --start YYYY-MM-DD --end YYYY-MM-DD
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import zlib
from urllib.parse import urlencode

import numpy as np
from agent.mappings import MERRA2_COLUMN_VARIABLES, canonical_variable
from . import fields, grid

UTC = timezone.utc
MAX_HOURS = 168


def root():
    return Path(os.environ.get('HOURLY_FIELDS_DIR', '/data/hourly'))


def variable_name(value):
    value = canonical_variable(value)
    if value not in MERRA2_COLUMN_VARIABLES:
        raise grid.BadGridRequestError('Unsupported hourly aerosol variable.')
    return value


def day_path(var, day):
    return root() / variable_name(var) / (date.fromisoformat(day).isoformat() + '.zarr')


def stamps(day):
    start = datetime.combine(date.fromisoformat(day), datetime.min.time(), UTC)
    return [start + timedelta(hours=h, minutes=30) for h in range(24)]


def coverage(var):
    out = set()
    for p in (root() / variable_name(var)).glob('*.zarr'):
        try:
            meta = json.loads((p / 'complete.json').read_text())
            if meta['version'] == 1 and meta['hours'] == 24:
                out.add(date.fromisoformat(p.stem))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return out


def latest_window(var, hours=48):
    if not 1 <= hours <= MAX_HOURS:
        raise grid.BadGridRequestError('Hourly windows must contain 1–168 frames.')
    days = coverage(var)
    for end_day in sorted(days, reverse=True):
        end = stamps(end_day.isoformat())[-1]
        start = end - timedelta(hours=hours - 1)
        needed = { (start + timedelta(hours=h)).date() for h in range(hours) }
        if needed <= days:
            return start, end
    raise grid.IndexNotBuiltError('No complete recent hourly window is stored for this variable. Run the hourly ingestion command first.')


@contextmanager
def writer_lock(name=".writer.lock"):
    root().mkdir(parents=True, exist_ok=True)
    with (root() / name).open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def publish(var, day, data, times):
    """Caller holds writer_lock. Incomplete days never enter coverage."""
    import zarr
    expected = stamps(day)
    actual = [grid._parse_time(str(t).replace(' ', 'T'), 'hourly timestamp') for t in times]
    if actual != expected:
        raise grid.GridFetchError(f'{day}: expected all 24 hourly centers, 00:30–23:30 UTC.')
    arr = np.asarray(data, dtype='<f4')
    if arr.shape != (24, fields.NLAT, fields.NLON):
        raise grid.GridFetchError(f'Unexpected hourly array shape: {arr.shape}')
    arr = arr.copy()
    arr[~np.isfinite(arr)] = np.nan
    target = day_path(var, day)
    if (target / 'complete.json').exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='.day-', dir=target.parent))
    try:
        store = zarr.open_array(str(tmp), mode='w', shape=arr.shape,
                                chunks=(6, 91, 144), dtype='<f4')
        store[:] = arr
        (tmp / 'complete.json').write_text(json.dumps({'version': 1, 'hours': 24}))
        if target.exists():
            shutil.rmtree(target)  # recovery of a previously incomplete local day
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)


def ensure_day(var, day):
    """Materialize one indexed variable-day, reusable across all viewports."""
    var = variable_name(var)
    if date.fromisoformat(day) in coverage(var):
        return
    with writer_lock('.archive.lock'):
        if date.fromisoformat(day) in coverage(var):
            return
        first, last = stamps(day)[0], stamps(day)[-1]
        manifest = grid._load_manifest(grid._index_path())
        refs = grid._refs_for_window(manifest, first, last)
        datasets = grid._open_datasets(refs, grid._default_target_options(refs))
        try:
            da = grid.slice_variable(datasets[0], var, (-180, -90, 180, 90), first, last)
            da = da.transpose('time', 'lat', 'lon').sortby('lat').sortby('lon')
            if not (np.allclose(da.lat.values, fields._latitudes()) and
                    np.allclose(da.lon.values, fields._longitudes())):
                raise grid.GridFetchError('Unexpected MERRA-2 coordinate grid.')
            da = da.compute()
            with writer_lock():
                publish(var, day, da.values, da.time.values)
        finally:
            for ds in datasets:
                ds.close()


def coordinates(bbox):
    w, s, e, n = grid._parse_bbox(bbox)
    if w == e:
        raise grid.BadGridRequestError("Bounding box west and east must differ.")
    # Expand small city viewports to native cell centers; do not invent resolution.
    s, n = max(-90, np.floor(s * 2) / 2), min(90, np.ceil(n * 2) / 2)
    w, e = max(-180, np.floor(w / .625) * .625), min(180, np.ceil(e / .625) * .625)
    yi, lats = fields._lat_index(s, n)
    xi, lons = fields._lon_index(w, e)
    return yi, xi, lats.tolist(), lons.tolist()


def window(t0, t1):
    start, end = grid._parse_time(t0, 't0'), grid._parse_end_time(t1, 't1')
    if end < start or end - start > timedelta(hours=MAX_HOURS):
        raise grid.BadGridRequestError('Choose an hourly window of at most seven days.')
    first = start.replace(minute=30, second=0, microsecond=0)
    if first < start:
        first += timedelta(hours=1)
    times = []
    while first <= end:
        times.append(first)
        first += timedelta(hours=1)
    if not times or len(times) > MAX_HOURS:
        raise grid.BadGridRequestError('Window selects no hours or exceeds 168 frames.')
    return times


def frame_manifest(variable, bbox, t0, t1):
    var = variable_name(variable)
    times = window(t0, t1)
    _, _, lats, lons = coordinates(bbox)
    cached = coverage(var)
    needed = {t.date() for t in times}
    if not needed <= cached:
        index = grid._load_manifest(grid._index_path())
        # Check every date, rather than allowing sparse manifest windows.
        for d in needed - cached:
            grid._refs_for_window(index, stamps(d.isoformat())[0], stamps(d.isoformat())[-1])
    batches = []
    for offset in range(0, len(times), 6):
        subset = times[offset:offset + 6]
        query = urlencode({'variable': var, 'bbox': bbox,
                           't0': subset[0].isoformat(), 't1': subset[-1].isoformat()})
        batches.append({'offset': offset, 'count': len(subset), 'url': '/frames/batch?' + query})
    return {'version': 1, 'variable': var, 'units': 'AOD (unitless)',
            'encoding': 'float32-le', 'missing': 'NaN', 'order': 'time,lat,lon',
            'lats': lats, 'lons': lons, 'nx': len(lons), 'ny': len(lats),
            'timestamps': [t.isoformat() for t in times], 'batches': batches,
            'range': {'vmin': 0, 'vmax': 1}, 'aggregation': 'hourly',
            'time_start': times[0].isoformat(), 'time_end': times[-1].isoformat()}


def _remote_frames(var, times, lats, lons):
    """Select native coordinates and hours lazily before any aerosol reads."""
    manifest = grid._load_manifest(grid._index_path())
    frames = []
    for day in sorted({t.date() for t in times}):
        selected = [t for t in times if t.date() == day]
        refs = grid._refs_for_window(manifest, selected[0], selected[-1])
        datasets = grid._open_datasets(refs, grid._default_target_options(refs))
        try:
            if len(datasets) != 1:
                raise grid.GridFetchError('Expected one hourly granule per date.')
            # Explicit coordinate selection preserves wrapped longitude order and
            # fails on missing hours/cells instead of silently returning a gap.
            da = datasets[0][var].sel(
                time=np.array([t.replace(tzinfo=None) for t in selected], dtype='datetime64[ns]'),
                lat=lats, lon=lons).transpose('time', 'lat', 'lon')
            data = np.asarray(da.compute(scheduler='threads', num_workers=4).values, dtype='<f4')
            if data.shape != (len(selected), len(lats), len(lons)):
                raise grid.GridFetchError('Unexpected regional hourly shape.')
            data = data.copy()
            data[~np.isfinite(data)] = np.nan
            frames.extend(data)
        except grid.GridFetchError:
            raise
        except Exception as exc:
            raise grid.GridFetchError(f'Regional hourly read failed: {exc}') from exc
        finally:
            for ds in datasets:
                ds.close()
    return frames


def frame_batch(variable, bbox, t0, t1):
    import zarr
    var = variable_name(variable)
    times = window(t0, t1)
    if len(times) > 6:
        raise grid.BadGridRequestError('A batch may contain at most six hours.')
    yi, xi, lats, lons = coordinates(bbox)
    # Cache the normalized native-cell selection, not the user's bbox spelling.
    key = hashlib.sha256(json.dumps([1, var, [t.isoformat() for t in times],
                                     lats, lons]).encode()).hexdigest()
    cache = root() / 'batches'
    target = cache / (key + '.gz')

    def read_cached():
        try:
            payload = target.read_bytes()
            if len(gzip.decompress(payload)) != len(times) * len(lats) * len(lons) * 4:
                return None
            os.utime(target, None)
            return payload
        except (OSError, EOFError, zlib.error):
            return None

    hit = read_cached()
    if hit is not None:
        return hit
    # Keep warm reads independent of the archive lock and protect against eviction.
    frames = {}
    with writer_lock():
        stored = coverage(var)
        for t in times:
            if t.date() in stored:
                path = day_path(var, t.date().isoformat())
                arr = zarr.open_array(str(path), mode='r')
                frames[t] = np.asarray(arr.oindex[t.hour, yi, xi], dtype='<f4')
                os.utime(path, None)
    missing = [t for t in times if t not in frames]
    if not missing:
        return gzip.compress(np.stack([frames[t] for t in times]).astype('<f4').tobytes(), compresslevel=1, mtime=0)
    # Bound concurrent archive jobs on small hosts; warm cache hits bypass this.
    with writer_lock('.archive.lock'):
        hit = read_cached()
        if hit is not None:
            return hit
        frames.update(zip(missing, _remote_frames(var, missing, lats, lons)))
        payload = gzip.compress(np.stack([frames[t] for t in times]).astype('<f4').tobytes(), compresslevel=1, mtime=0)
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cache, prefix='.batch-', delete=False) as f:
            tmp = Path(f.name)
            try:
                f.write(payload)
                f.close()
                os.replace(tmp, target)
            finally:
                tmp.unlink(missing_ok=True)
        cap = max(0, int(os.environ.get('HOURLY_BATCH_CACHE_MAX_MB', '256'))) * 1024**2
        entries = sorted((p.stat().st_mtime, p.stat().st_size, p) for p in cache.glob('*.gz'))
        total = sum(size for _, size, _ in entries)
        for _, size, path in entries:
            if total <= cap:
                break
            path.unlink(missing_ok=True)
            total -= size
        return payload


def prune(retain_days=7, max_mb=2048, pinned=()):
    """Keep recent published days and pinned events; evict other days by mtime.

    The byte cap applies to historical cache, not the protected rolling window.
    """
    with writer_lock():
        try:
            saved_pins = json.loads((root() / 'pins.json').read_text())
        except FileNotFoundError:
            saved_pins = []
        pinned = set(pinned) | {date.fromisoformat(d) for d in saved_pins}
        candidates = []
        for var in MERRA2_COLUMN_VARIABLES:
            days = sorted(coverage(var), reverse=True)
            protected = set(days[:retain_days]) | set(pinned)
            for d in days:
                if d not in protected:
                    p = day_path(var, d.isoformat())
                    size = sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
                    candidates.append((p.stat().st_mtime, size, p))
        total = sum(item[1] for item in candidates)
        for _, size, p in sorted(candidates):
            if total <= max_mb * 1024**2:
                break
            shutil.rmtree(p)
            total -= size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start')
    parser.add_argument('--end')
    parser.add_argument('--events', nargs='*', default=[], help='Catalog IDs whose first 48 hours to ingest and pin')
    parser.add_argument('--variables', nargs='+', default=sorted(MERRA2_COLUMN_VARIABLES))
    parser.add_argument('--retain-days', type=int, default=7)
    parser.add_argument('--historical-cache-mb', type=int, default=2048)
    args = parser.parse_args()
    if args.retain_days < 2 or args.historical_cache_mb < 0:
        parser.error('Require retain-days >= 2 and nonnegative cache size.')
    if bool(args.start) != bool(args.end):
        parser.error('Supply both --start and --end, or neither.')
    tasks = []
    pins = set()
    if args.events:
        from .plume_router import events
        catalog = {e['id']: e for e in events()}
        for event_id in args.events:
            if event_id not in catalog:
                parser.error('Unknown catalog ID: ' + event_id)
            event = catalog[event_id]
            start = date.fromisoformat(event['time_start'][:10])
            var = next(p['variable'] for p in event['suggested_plans'] if p['level'] == 'column')
            for day in (start, start + timedelta(days=1)):
                tasks.append((var, day))
                pins.add(day.isoformat())
    if args.start:
        start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
        if start > end:
            parser.error('Require start <= end.')
        while start <= end:
            tasks.extend((v, start) for v in args.variables)
            start += timedelta(days=1)
    elif not args.events:
        manifest = grid._load_manifest(grid._index_path())
        days = sorted(date.fromisoformat(d) for d in manifest.get('files', {}))
        if not days:
            parser.error('No indexed dates. Build the kerchunk index first.')
        tasks.extend((v, d) for d in days[-args.retain_days:] for v in args.variables)
    for var, day in tasks:
        ensure_day(var, day.isoformat())
        print(f'Stored {var} {day}', flush=True)
    if pins:
        with writer_lock():
            path = root() / 'pins.json'
            existing = set(json.loads(path.read_text())) if path.exists() else set()
            tmp = root() / '.pins.tmp'
            tmp.write_text(json.dumps(sorted(existing | pins)))
            os.replace(tmp, path)
    prune(args.retain_days, args.historical_cache_mb)


if __name__ == '__main__':
    main()
