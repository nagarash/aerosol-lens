"""Hourly routing eval with fixed coverage and no NASA calls.

Offline: python -m backend.eval_hourly
Live (one model call at most per ambiguous case):
  python -m backend.eval_hourly --model openrouter/YOUR_MODEL --live
Live mode incurs provider charges and requires its key. Reports JSON for CI.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import time
from unittest.mock import patch

from . import hourly, plume_router

CASES = [
    ('multi_place', 'is Delhi or Beijing dustier today', 'clarify', None),
    ('state_default', 'dust over Georgia this week', 'latest', 'Georgia'),
    ('transport_future', 'show Saharan dust moving toward Florida over the next few days', 'clarify', None),
    ('event_indirect', 'that huge dust cloud everyone called Godzilla back in 2020', 'event', None),
    ('figurative', 'why was the sky orange over LA that one September', 'clarify', None),
    ('event_conflict', 'the volcanic eruption near Tonga last year', 'clarify', None),
    ('health', 'is it safe to go for a run in Beijing right now', 'clarify', None),
    ('comparison', 'compare wildfire smoke in LA to dust storms in Phoenix last month', 'clarify', None),
    ('global', 'show me global dust levels right now', 'latest', 'Global'),
    ('here_health', 'is the air near me safe to breathe today', 'clarify', None),
    ('extended', 'dust storm over the Atacama Desert last week', 'latest', 'Atacama Desert'),
    ('recent', 'smoke over California a few days ago', 'latest', 'California'),
    ('historical', 'dust over Arizona 2020-06-14 to 2020-06-15', 'historical', 'Arizona'),
    ('country_override', 'dust over Georgia country now', 'latest', 'Georgia country'),
]


def evaluate(model_call):
    rows = []
    latest = datetime(2026,8,25,23,30,tzinfo=timezone.utc)
    for case_id, question, expected, place in CASES:
        calls = []
        def call(messages):
            calls.append(messages)
            return model_call(messages)
        start = time.perf_counter()
        outcome, detail = 'FAIL', ''
        with patch.object(hourly, 'latest_window', return_value=(latest-timedelta(hours=47), latest)):
            try:
                plan, info = plume_router.route(question, '2026-09-26', model_call=call)
                ok = expected != 'clarify' and (place is None or plan.place_name == place)
                if expected == 'latest':
                    ok &= plan.time_end == latest and plan.time_end-plan.time_start == timedelta(hours=47)
                elif expected in ('historical', 'event'):
                    ok &= plan.time_start.date().isoformat() == '2020-06-14' and plan.variable == 'DUEXTTAU'
                outcome = 'PASS' if ok else 'FAIL'
                detail = plan.model_dump(mode='json')
            except plume_router.Clarification as exc:
                outcome = 'PASS' if expected == 'clarify' else 'FAIL'
                detail = str(exc)
            except Exception as exc:
                detail = type(exc).__name__ + ': ' + str(exc)
        if len(calls) > 1:
            outcome = 'FAIL'
        rows.append({'id':case_id, 'outcome':outcome, 'model_calls':len(calls),
                     'elapsed_ms':round((time.perf_counter()-start)*1000,2), 'detail':detail})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--model')
    args = parser.parse_args()
    if args.live and not args.model:
        parser.error('--live requires --model')
    usage = []
    def model_call(messages):
        if args.live:
            import litellm
            response = litellm.completion(model=args.model, messages=messages,
                response_format={'type':'json_object'}, max_tokens=400, timeout=10, num_retries=0)
            usage.append(response.usage.model_dump() if response.usage else {})
            return response.choices[0].message.content or ''
        question = json.loads(messages[-1]['content'])['question'].lower()
        if 'godzilla' in question:
            return '{"event_id":"sahara-dust-godzilla-2020","time_mode":"historical"}'
        return '{"clarification":"Please specify the event year or exact dates."}'
    rows = evaluate(model_call)
    print(json.dumps({'mode':'live' if args.live else 'offline-stub', 'results':rows, 'usage':usage}, indent=2))
    raise SystemExit(1 if any(r['outcome']=='FAIL' for r in rows) else 0)


if __name__ == '__main__':
    main()
