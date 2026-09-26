"""Hourly MERRA-2 storage and six-hour binary frame batches.

Each immutable variable/day Zarr is published by atomic rename after validation.
Daily means are never used. Cold indexed days are fetched one variable at a time
under a process-wide file lock, bounding archive concurrency and peak memory.
Run: python -m backend.hourly --start YYYY-MM-DD --end YYYY-MM-DD
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
import fcntl
import gzip
import json
import os
from pathlib import Path
import shutil
import tempfile
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
    w, s, e, n = _expanded_bbox(bbox)
    yi, lats = fields._lat_index(s, n)
    xi, lons = fields._lon_index(w, e)
    return yi, xi, lats.tolist(), lons.tolist()


def _expanded_bbox(bbox):
    """Parse and expand a bbox to native MERRA-2 cell centers.

    Shared by coordinates() and the patch store so a cropped fetch selects
    exactly the cells a later read expects.
    """
    w, s, e, n = grid._parse_bbox(bbox)
    if w == e:
        raise grid.BadGridRequestError("Bounding box west and east must differ.")
    # Expand small city viewports to native cell centers; do not invent resolution.
    s, n = max(-90, np.floor(s * 2) / 2), min(90, np.ceil(n * 2) / 2)
    w, e = max(-180, np.floor(w / .625) * .625), min(180, np.ceil(e / .625) * .625)
    return w, s, e, n


# --------------------------------------------------------------------------
# Bbox-cropped patch store (cold-path fast lane)
#
# A cold historical query for one variable and one viewport does not need
# the full 120MB global day: NASA's NetCDF chunks are (1h, 91lat, 144lon),
# so kerchunk range reads fetch only the chunks intersecting the bbox
# (typically ~1MB for a city-sized query). Patches are keyed by exact
# expanded bbox; the full global day remains the canonical reusable unit
# for the hot rolling window.

def patch_key(bbox):
    """Stable hash of an expanded bbox for patch filenames."""
    import hashlib
    w, s, e, n = _expanded_bbox(bbox)
    norm = f"{w:.4f},{s:.4f},{e:.4f},{n:.4f}"
    return hashlib.sha256(norm.encode()).hexdigest()[:16]


def patch_path(var, day, key):
    return root() / variable_name(var) / 'patches' / f"{day}_{key}.zarr"


def find_patch(var, day, bbox):
    """Return the stored patch for this exact (variable, day, bbox), or None."""
    p = patch_path(var, day, patch_key(bbox))
    try:
        meta = json.loads((p / 'complete.json').read_text())
    except (OSError, ValueError):
        return None
    if meta.get('version') == 1 and meta.get('hours') == 24:
        return p
    return None


def publish_patch(var, day, bbox, da):
    """Atomically publish a bbox-cropped variable-day. Caller holds writer_lock.

    The patch grid must equal coordinates(bbox): same cells, same order,
    so reads need no remapping.
    """
    import zarr
    var = variable_name(var)
    day = date.fromisoformat(day).isoformat()
    _, _, lats, lons = coordinates(bbox)
    expected = stamps(day)
    actual = [grid._parse_time(str(t).replace(' ', 'T'), 'hourly timestamp')
              for t in da.time.values]
    if actual != expected:
        raise grid.GridFetchError(f'{day}: expected all 24 hourly centers, 00:30–23:30 UTC.')
    arr = np.asarray(da.values, dtype='<f4')
    if arr.shape != (24, len(lats), len(lons)):
        raise grid.GridFetchError(f'Unexpected patch shape: {arr.shape}')
    if not (np.allclose(np.asarray(da.lat.values), lats) and
            np.allclose(np.asarray(da.lon.values), lons)):
        raise grid.GridFetchError('Patch grid does not match coordinates(bbox).')
    arr = arr.copy()
    arr[~np.isfinite(arr)] = np.nan
    target = patch_path(var, day, patch_key(bbox))
    if (target / 'complete.json').exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='.patch-', dir=target.parent))
    try:
        ny, nx = arr.shape[1], arr.shape[2]
        store = zarr.open_array(str(tmp), mode='w', shape=arr.shape,
                                chunks=(6, min(91, ny), min(144, nx)), dtype='<f4')
        store[:] = arr
        (tmp / 'complete.json').write_text(json.dumps({
            'version': 1, 'hours': 24,
            'bbox': list(_expanded_bbox(bbox)),
            'lats': lats, 'lons': lons}))
        if target.exists():
            shutil.rmtree(target)  # recovery of a previously incomplete patch
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    return target


def ensure_patch(var, day, bbox):
    """Materialize one variable-day cropped to bbox via kerchunk range reads.

    Only the NASA chunks intersecting the bbox are fetched (single variable).
    Returns the patch path.
    """
    var = variable_name(var)
    day = date.fromisoformat(day).isoformat()
    p = find_patch(var, day, bbox)
    if p:
        return p
    with writer_lock('.archive.lock'):
        p = find_patch(var, day, bbox)
        if p:
            return p
        w, s, e, n = _expanded_bbox(bbox)
        first, last = stamps(day)[0], stamps(day)[-1]
        manifest = grid._load_manifest(grid._index_path())
        refs = grid._refs_for_window(manifest, first, last)
        datasets = grid._open_datasets(refs, grid._default_target_options(refs))
        try:
            # Lazy: only chunks intersecting the bbox are fetched on compute().
            da = grid.slice_variable(datasets[0], var, (w, s, e, n), first, last)
            da = da.transpose('time', 'lat', 'lon').sortby('lat').sortby('lon')
            if len(da.time) != 24:
                raise grid.GridFetchError(
                    f'{day}: expected 24 hourly steps, got {len(da.time)}.')
            da = da.compute()
            with writer_lock():
                return publish_patch(var, day, bbox, da)
        finally:
            for ds in datasets:
                ds.close()


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
    # Dates servable from a patch don't need a manifest check either.
    uncached = {d for d in needed - cached
                if not find_patch(var, d.isoformat(), bbox)}
    if uncached:
        index = grid._load_manifest(grid._index_path())
        # Check every date, rather than allowing sparse manifest windows.
        for d in uncached:
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


def frame_batch(variable, bbox, t0, t1):
    import zarr
    var = variable_name(variable)
    times = window(t0, t1)
    if len(times) > 6:
        raise grid.BadGridRequestError('A batch may contain at most six hours.')
    yi, xi, _, _ = coordinates(bbox)
    # Resolve each date: full global day preferred, else a bbox-cropped patch
    # fetched with single-variable range reads (the cold fast lane).
    sources = {}  # date -> (path, is_patch)
    fetched = False
    for d in sorted({t.date() for t in times}):
        ds = d.isoformat()
        if d in coverage(var):
            sources[d] = (day_path(var, ds), False)
            continue
        p = find_patch(var, ds, bbox)
        if p is None:
            p = ensure_patch(var, ds, bbox)
            fetched = True
        sources[d] = (p, True)
    # Hold the retention lock during local reads, preventing eviction races.
    with writer_lock():
        frames = []
        for t in times:
            path, is_patch = sources[t.date()]
            arr = zarr.open_array(str(path), mode='r')
            if is_patch:
                # Patch grid already equals coordinates(bbox): no remapping.
                frames.append(np.asarray(arr[t.hour], dtype='<f4'))
            else:
                frames.append(np.asarray(arr.oindex[t.hour, yi, xi], dtype='<f4'))
            os.utime(path, None)
    payload = gzip.compress(np.stack(frames).astype('<f4').tobytes(), compresslevel=1, mtime=0)
    if fetched:
        prune(int(os.environ.get("HOURLY_RETAIN_DAYS", "7")),
              int(os.environ.get("HOURLY_CACHE_MAX_MB", "2048")))
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
            full = set(days)
            for d in days:
                if d not in protected:
                    p = day_path(var, d.isoformat())
                    size = sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
                    candidates.append((p.stat().st_mtime, size, p))
            # Patches share the historical budget. A patch made redundant by
            # a full global day is dropped outright.
            pdir = root() / var / 'patches'
            if pdir.is_dir():
                for p in pdir.glob('*.zarr'):
                    try:
                        pday = date.fromisoformat(p.stem.split('_')[0])
                    except ValueError:
                        continue
                    if pday in full:
                        shutil.rmtree(p)
                        continue
                    if pday in protected or pday in pinned:
                        continue
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
