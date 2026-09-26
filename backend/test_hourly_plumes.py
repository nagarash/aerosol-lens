"""Offline hourly pipeline acceptance tests; no provider or NASA calls."""
import gzip
import json
from datetime import datetime, timedelta, timezone, date
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend import hourly, plume_router as router, geocode, app as api
from backend.grid import IndexNotBuiltError, GridFetchError, BadGridRequestError

UTC = timezone.utc
LATEST = datetime(2026, 8, 25, 23, 30, tzinfo=UTC)


@pytest.fixture
def available():
    with patch.object(hourly, 'latest_window', return_value=(LATEST-timedelta(hours=47), LATEST)):
        yield


def never_model(_):
    pytest.fail('Deterministic query must not call a model')


@pytest.mark.parametrize('q', ['dust over Arizona', 'dust over Arizona right now',
                               'dust over Arizona a few days ago', 'smoke over California yesterday'])
def test_recent_48_hours(q, available):
    plan, info = router.route(q, '2026-09-26', model_call=never_model)
    assert len(hourly.window(plan.time_start.isoformat(), plan.time_end.isoformat())) == 48
    assert plan.time_end == LATEST
    assert info['routing'] == 'deterministic'
    assert info['time_mode'] == 'latest'


def test_historical_dates_do_not_shift(available):
    plan, info = router.route('dust over Arizona 2020-06-14 to 2020-06-15', '2026-09-26', model_call=never_model)
    assert plan.time_start.year == 2020
    assert len(hourly.window(plan.time_start.isoformat(), plan.time_end.isoformat())) == 48
    assert info['time_mode'] == 'historical'


def test_event_catalog_and_explicit_override():
    p, _ = router.route('show godzilla dust', '2026-09-26', model_call=never_model)
    assert p.time_start.date() == date(2020, 6, 14)
    assert p.variable == 'DUEXTTAU'
    p, _ = router.route('godzilla dust 2020-06-20 to 2020-06-21', '2026-09-26', model_call=never_model)
    assert p.time_start.date() == date(2020, 6, 20)


def test_wrapped_event_and_nested_places(available):
    p, _ = router.route('hunga tonga eruption', '2026-09-26', model_call=never_model)
    assert p.bbox[0] > p.bbox[2]
    assert router.places_in('dust over West Virginia') == ['West Virginia']
    assert router.places_in('dust over Washington DC') == ['Washington, D.C.']
    p, _ = router.route('dust moving from Alaska toward California', '2026-09-26', model_call=never_model)
    assert p.bbox[0] > p.bbox[2]


def test_qualified_georgia(available):
    p, _ = router.route('dust over Georgia', '2026-09-26', model_call=never_model)
    assert p.bbox[0] < -80
    p, _ = router.route('dust over Georgia country', '2026-09-26', model_call=never_model)
    assert p.bbox[0] > 40


@pytest.mark.parametrize('q', ['compare dust over Delhi and Beijing', 'dust over Arizona tomorrow',
                               'is it safe to run in Beijing', 'show PM2.5 over Delhi'])
def test_unsupported_requests(q, available):
    with pytest.raises(router.Clarification):
        router.route(q, '2026-09-26', model_call=never_model)


def test_model_once_and_invalid_output_not_retried(available):
    calls = []
    def model(messages):
        calls.append(messages)
        return json.dumps({'places': ['California'], 'variable': 'BCEXTTAU',
                           'time_mode': 'historical', 'start': '2020-09-09', 'end': '2020-09-10'})
    p, info = router.route('that orange sky in California in September 2020', '2026-09-26', model_call=model)
    assert len(calls) == 1 and info['routing'] == 'model'
    assert p.time_start.year == 2020
    calls.clear()
    def bad(messages):
        calls.append(messages)
        return '{invalid'
    with pytest.raises(ValueError):
        router.route('orange sky somewhere', '2026-09-26', model_call=bad)
    assert len(calls) == 1


def test_unknown_event_must_not_be_invented():
    with pytest.raises(router.Clarification):
        router.route('the mysterious cloud', '2026-09-26', model_call=lambda _: '{"event_id":"made-up"}')


def test_coverage_requires_contiguous_days(monkeypatch, tmp_path):
    monkeypatch.setenv('HOURLY_FIELDS_DIR', str(tmp_path))
    for day in ['2020-01-01', '2020-01-03']:
        p = hourly.day_path('DUEXTTAU', day)
        p.mkdir(parents=True)
        (p/'complete.json').write_text('{"version":1,"hours":24}')
    with pytest.raises(IndexNotBuiltError):
        hourly.latest_window('DUEXTTAU')
    assert hourly.coverage('BCEXTTAU') == set()
    incomplete = hourly.day_path('DUEXTTAU', '2020-01-02')
    incomplete.mkdir()
    assert date(2020,1,2) not in hourly.coverage('DUEXTTAU')


def test_store_binary_and_http(monkeypatch, tmp_path):
    monkeypatch.setenv('HOURLY_FIELDS_DIR', str(tmp_path))
    monkeypatch.setenv('HOURLY_PLUMES_ENABLED', '1')
    data = np.broadcast_to(np.arange(24, dtype='float32')[:,None,None], (24,361,576)).copy()
    data[0, 200, 400] = np.nan
    for day in ['2020-01-01', '2020-01-02']:
        with hourly.writer_lock():
            hourly.publish('DUEXTTAU', day, data, hourly.stamps(day))
    with patch.object(hourly.grid, '_open_datasets', side_effect=AssertionError('Warm path must not call NASA')):
        client = TestClient(api.app)
        res = client.post('/ask', json={'question': 'dust over Arizona now', 'reference_date':'2026-09-26'})
        assert res.status_code == 200, res.text
        assert res.json()['resolution']['routing'] == 'deterministic'
        manifest_response = client.get(res.json()['data_url'])
        assert manifest_response.status_code == 200, manifest_response.text
        manifest = manifest_response.json()
        assert len(manifest['timestamps']) == 48
        assert len(manifest['batches']) == 8
        response = client.get(manifest['batches'][0]['url'])
        assert response.status_code == 200, response.text
        # httpx transparently decompresses Content-Encoding: gzip.
        array = np.frombuffer(response.content, dtype='<f4').reshape(6,manifest['ny'],manifest['nx'])
        assert np.all(array[0] == 0) and np.all(array[5] == 5)
        raw = hourly.frame_batch('DUEXTTAU', '70,10,70.625,10.5',
                                 '2020-01-01T00:30:00Z', '2020-01-01T00:30:00Z')
        assert np.isnan(np.frombuffer(gzip.decompress(raw), dtype='<f4')[0])
    assert api.ask_hourly(api.AskRequest(question='dust over Arizona 2020-01-01 to 2020-01-02')).plan.time_start.year == 2020


def test_incomplete_day_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv('HOURLY_FIELDS_DIR', str(tmp_path))
    with pytest.raises(GridFetchError):
        with hourly.writer_lock():
            hourly.publish('DUEXTTAU', '2020-01-01', np.zeros((1,361,576)), hourly.stamps('2020-01-01')[:1])
    assert not hourly.coverage('DUEXTTAU')


def test_manifest_rejects_missing_historical_day(monkeypatch, tmp_path):
    monkeypatch.setenv('HOURLY_FIELDS_DIR', str(tmp_path))
    with patch.object(hourly.grid, '_load_manifest', return_value={'files':{'2020-01-01':'a.json'}}):
        with pytest.raises(BadGridRequestError):
            hourly.frame_manifest('DUEXTTAU','-10,0,10,10','2020-01-01','2020-01-02')


def test_antimeridian_union():
    assert geocode.union_boxes([[170,-20,-170,20], [175,-10,-160,10]]) == [170,-20,-160,20]
    assert geocode.union_boxes([[-180,-90,180,90], [10,10,20,20]]) == [-180,-90,180,90]


def test_named_date_range_is_local():
    p, _ = router.route('smoke over California September 9-10, 2020', '2026-09-26', model_call=never_model)
    assert p.time_start.date() == date(2020,9,9)
    assert p.time_end.date() == date(2020,9,10)


def test_here_without_model(available):
    p, _ = router.route('dust near me', '2026-09-26', user_location='33.5,-112', model_call=never_model)
    assert p.place_name == 'Your area'
    assert p.bbox == [-113,32.5,-111,34.5]


def test_relative_duration_has_no_extra_calendar_day():
    with patch.object(hourly, 'coverage', return_value={date(2026,8,24),date(2026,8,25)}):
        p, _ = router.route('dust over Arizona last 2 days', '2026-09-26', model_call=never_model)
    assert len(hourly.window(p.time_start.isoformat(), p.time_end.isoformat())) == 48


def test_cold_day_preserves_hours_and_is_reused(monkeypatch, tmp_path):
    import xarray as xr
    monkeypatch.setenv('HOURLY_FIELDS_DIR', str(tmp_path))
    data = np.broadcast_to(np.arange(24, dtype='float32')[:,None,None], (24,361,576))
    ds = xr.Dataset({'DUEXTTAU': (('time','lat','lon'), data)}, coords={
        'time': np.array([t.replace(tzinfo=None) for t in hourly.stamps('2020-01-01')], dtype='datetime64[ns]'),
        'lat':hourly.fields._latitudes(), 'lon':hourly.fields._longitudes()})
    with patch.object(hourly.grid, '_load_manifest', return_value={'files':{'2020-01-01':'local.json'}}), \
         patch.object(hourly.grid, '_default_target_options', return_value={}), \
         patch.object(hourly.grid, '_open_datasets', return_value=[ds]) as opened:
        one = hourly.frame_batch('DUEXTTAU','0,0,1,1','2020-01-01T00:30:00Z','2020-01-01T05:30:00Z')
        two = hourly.frame_batch('DUEXTTAU','0,0,1,1','2020-01-01T06:30:00Z','2020-01-01T11:30:00Z')
        assert opened.call_count == 1
        assert np.frombuffer(gzip.decompress(one),dtype='<f4')[0] == 0
        assert np.frombuffer(gzip.decompress(two),dtype='<f4')[0] == 6


def test_retention_keeps_recent_and_pinned(monkeypatch, tmp_path):
    monkeypatch.setenv('HOURLY_FIELDS_DIR', str(tmp_path))
    for i in range(1,5):
        p = hourly.day_path('DUEXTTAU',f'2020-01-0{i}')
        p.mkdir(parents=True)
        (p/'complete.json').write_text('{"version":1,"hours":24}')
    (tmp_path/'pins.json').write_text('["2020-01-01"]')
    hourly.prune(retain_days=2, max_mb=0)
    assert hourly.coverage('DUEXTTAU') == {date(2020,1,1),date(2020,1,3),date(2020,1,4)}


def test_daily_mean_store_cannot_serve_subday():
    from backend import grid
    with patch.object(grid, '_require_geo_deps'), \
         patch.object(grid.fields_store, 'available', return_value=True), \
         patch.object(grid.fields_store, 'coverage', return_value={'2020-01-01'}), \
         patch.object(grid, '_load_manifest', side_effect=IndexNotBuiltError('hourly archive required')):
        with pytest.raises(IndexNotBuiltError, match='hourly archive required'):
            grid.get_grid('merra2','DUEXTTAU','0,0,10,10','2020-01-01T01:00:00Z','2020-01-01T02:00:00Z','hourly')


def test_model_cannot_replace_historical_reference_with_latest(available):
    with pytest.raises(router.Clarification, match='historical'):
        router.route('dust in California in 2020', '2026-09-26', model_call=lambda _: '{"places":["California"],"time_mode":"latest"}')


def test_model_event_conflicting_year_rejected():
    with pytest.raises(router.Clarification, match='conflict'):
        router.route('eruption near Tonga last year', '2026-09-26', model_call=lambda _: '{"event_id":"hunga-tonga-2022"}')


def test_interpretation_cache_reuses_only_valid_output(monkeypatch):
    from types import SimpleNamespace
    api._cached_hourly_interpretation.cache_clear()
    messages = '[{"role":"user","content":"test"}]'
    valid = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"places":["California"]}'))])
    with patch.object(api.litellm, 'completion', return_value=valid) as completion:
        api._cached_hourly_interpretation('mock/test', messages)
        api._cached_hourly_interpretation('mock/test', messages)
        assert completion.call_count == 1
        assert completion.call_args.kwargs['num_retries'] == 0
    api._cached_hourly_interpretation.cache_clear()
    invalid = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='invalid'))])
    with patch.object(api.litellm, 'completion', return_value=invalid) as completion:
        for _ in range(2):
            with pytest.raises(ValueError):
                api._cached_hourly_interpretation('mock/test', messages)
        assert completion.call_count == 2
    api._cached_hourly_interpretation.cache_clear()
