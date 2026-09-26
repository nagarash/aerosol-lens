"""Local-first plume interpretation; at most one optional general-model call."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from agent.query_plan import QueryPlan, StyleSpec
from agent.mappings import MERRA2_COLUMN_VARIABLES
from . import geocode, hourly

UTC = timezone.utc


class Clarification(ValueError):
    pass


class Interpretation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    places: list[str] = Field(default_factory=list, max_length=4)
    variable: str = 'TOTEXTTAU'
    time_mode: Literal['latest', 'historical'] = 'latest'
    start: str | None = None
    end: str | None = None
    duration_hours: int = Field(default=48, ge=1, le=168)
    event_id: str | None = None
    relationship: Literal['single', 'transport', 'comparison'] = 'single'
    clarification: str | None = None


@lru_cache(maxsize=1)
def events():
    p = Path(__file__).resolve().parent.parent / 'data/event_catalog.sample.json'
    return json.loads(p.read_text())['events']


@lru_cache(maxsize=1)
def matcher():
    import ahocorasick
    out = ahocorasick.Automaton()
    for alias, canonical in geocode.name_variants().items():
        out.add_word(alias, (alias, canonical))
    out.make_automaton()
    return out


def places_in(question):
    q = question.lower()
    hits = []
    for end, (alias, canonical) in matcher().iter(q):
        start, stop = end - len(alias) + 1, end + 1
        if (start and (q[start-1].isalnum() or q[start-1] == '_')) or (stop < len(q) and (q[stop].isalnum() or q[stop] == '_')):
            continue
        hits.append((start, stop, canonical))
    # Longest overlapping mention wins, without discarding separate locations.
    chosen = []
    for hit in sorted(hits, key=lambda h: -(h[1] - h[0])):
        if not any(hit[0] < other[1] and other[0] < hit[1] for other in chosen):
            chosen.append(hit)
    return list(dict.fromkeys(h[2] for h in sorted(chosen)))


SPECIES = {
    'DUEXTTAU': r'\b(dust|sandstorm|ash)\b',
    'BCEXTTAU': r'\b(smoke|wildfire|wildfires|black carbon)\b',
    'OCEXTTAU': r'\borganic carbon\b',
    'SUEXTTAU': r'\b(sulfate|volcanic|eruption)\b',
    'SSEXTTAU': r'\bsea salt\b',
}


def explicit_window(q):
    dates = re.findall(r'\b\d{4}-\d{2}-\d{2}\b', q)
    try:
        if dates:
            if len(dates) > 2:
                raise Clarification('Specify one date or one start/end date range.')
            first = datetime.fromisoformat(dates[0]).replace(tzinfo=UTC)
            last = datetime.fromisoformat(dates[-1]).replace(tzinfo=UTC)
        else:
            months = ('january february march april may june july august september october november december').split()
            names = '|'.join(months + [m[:3] for m in months])
            match = re.search(rf'\b({names})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:\s*(?:–|-|to)\s*(\d{{1,2}}))?,?\s+(\d{{4}})\b', q.lower())
            if not match:
                return None
            mon = next(i+1 for i,m in enumerate(months) if m.startswith(match[1]))
            first = datetime(int(match[4]), mon, int(match[2]), tzinfo=UTC)
            last = datetime(int(match[4]), mon, int(match[3] or match[2]), tzinfo=UTC)
        last += timedelta(days=1) - timedelta(seconds=1)
        if last < first:
            raise Clarification('The end date must follow the start date.')
        return first, last
    except ValueError as exc:
        raise Clarification('Please provide valid calendar dates in YYYY-MM-DD format.') from exc


def interpret_local(question):
    q = question.lower()
    if re.search(r'\b(aqi|pm2\.?5|pm10|safe to (?:breathe|run|go for a run)|safe.*(?:outside|exercise)|air quality|health|breathing|allergies)\b', q):
        raise Clarification('This viewer shows historical column aerosols, not breathing safety or surface air quality.')
    if re.search(r'\b(compare|versus|vs\.?|dustier|cleaner)\b', q) or re.search(r'\bor\b', q):
        raise Clarification('Choose one plume or a transport path; side-by-side comparisons are not supported yet.')
    if re.search(r'\b(tomorrow|forecast|next\s+(?:few|\d+)\s+days)\b', q):
        raise Clarification('MERRA-2 is historical. Ask for the latest available plume or historical dates.')
    matched_events = [e for e in events() if any(a.lower() in q for a in e['aliases'] + [e['name']])]
    species = [v for v, pattern in SPECIES.items() if re.search(pattern, q)]
    places = places_in(question)
    if re.search(r'\b(global|worldwide|whole world)\b', q):
        places = ['Global']
    transport = bool(re.search(r'\b(from|toward|towards|moving|transport|across)\b', q))
    dates = explicit_window(q)
    event = matched_events[0] if len(matched_events) == 1 else None
    if event and not dates:
        years = re.findall(r'\b(?:19|20)\d{2}\b', q)
        if (years and any(y != event['time_start'][:4] for y in years)) or re.search(r'\b(last year|years ago)\b', q):
            raise Clarification('The catalog event has different dates. Please specify the event year or an explicit date range.')
    # Unknown temporal language or indirect descriptions must reach the model.
    uncertain_time = bool(re.search(r'\b(19\d{2}|20\d{2}|january|february|march|april|may|june|july|august|september|october|november|december|last year|last month|ago|week|summer|winter|spring|autumn)\b', q))
    recent = bool(re.search(r'\b(now|today|yesterday|recent|latest|last\s+\d+\s+days|past\s+\d+\s+days|few days ago|this week|last week|past week)\b', q))
    understood_aerosol = bool(species or re.search(r'\b(aerosol|aerosols|aod|plume|haze)\b', q))
    if len(species) > 1 or len(matched_events) > 1 or (not event and (not places or not understood_aerosol or (uncertain_time and not dates and not recent))) or (len(places) > 1 and not transport and not event):
        return None
    # Bare month/day language and vague event references are not default-to-latest.
    if re.search(r'\b(event|orange sky|that time|years ago)\b', q) and not event and not dates:
        return None
    duration = 48
    match = re.search(r'\b(?:last|past|for)\s+(\d+)\s+(hours?|days?)\b', q)
    if match:
        duration = int(match[1]) * (24 if match[2].startswith('day') else 1)
        if not 1 <= duration <= 168:
            raise Clarification('Choose an hourly sequence of one hour to seven days.')
    return Interpretation(duration_hours=duration, places=places, variable=species[0] if species else 'TOTEXTTAU',
                          event_id=event['id'] if event else None,
                          time_mode='historical' if dates or event else 'latest',
                          start=dates[0].isoformat() if dates else None,
                          end=dates[1].isoformat() if dates else None,
                          relationship='transport' if transport and len(places) > 1 else 'single')


def model_messages(question, reference_date, user_location):
    return [
        {'role': 'system', 'content': (
            'Interpret a historical MERRA-2 plume visualization query. Return one JSON object with ONLY '
            'places (list of names), variable (MERRA-2 aerosol AOD code), time_mode (latest or historical), '
            'duration_hours (1–168, default 48), start/end (ISO UTC timestamps or null), event_id (catalog ID or null), '
            'relationship (single, transport, comparison), clarification (string or null). '
            'Never emit coordinates. Never answer health/AQI or forecasts: return clarification. '
            'Comparisons must remain comparison, never choose one side. Recent wording maps to latest '
            'published data. Explicit historical dates stay historical. Unknown or ambiguous events '
            'require clarification; never guess event dates. Use provided event IDs for catalog events. '
            'Use null dates for catalog events unless the user supplies specific dates. '
            'Place names must be grounded in the supplied candidates or catalog; return clarification '
            'if unclear. Georgia alone means the US state; Georgia country is a separate place. '
            'Smoke defaults to BCEXTTAU; dust DUEXTTAU; sulfate SUEXTTAU; '
            'organic carbon OCEXTTAU; sea salt SSEXTTAU; general aerosol TOTEXTTAU.\n'
            + json.dumps({'candidates': places_in(question), 'events': events()}) )},
        {'role': 'user', 'content': json.dumps({'question': question, 'reference_date': reference_date, 'user_location': user_location})},
    ]


def route(question, reference_date, user_location=None, model_call=None):
    location_bbox = None
    local_question = question
    if re.search(r'\b(here|near me|my area)\b', question.lower()):
        if not user_location:
            raise Clarification('Please supply a location for here or near me.')
        try:
            lat, lon = map(float, user_location.split(','))
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise Clarification('User coordinates are outside valid latitude/longitude bounds.')
            location_bbox = [max(-180, lon-1), max(-90, lat-1), min(180, lon+1), min(90, lat+1)]
            local_question = re.sub(r'\b(here|near me|my area)\b', 'Global', question, flags=re.I)
        except ValueError:
            if ',' in user_location:
                raise Clarification('User location must contain valid latitude,longitude or a known place name.')
            local_question = re.sub(r'\b(here|near me|my area)\b', user_location, question, flags=re.I)
    result = interpret_local(local_question)
    routing = 'deterministic'
    if result is None:
        if model_call is None:
            raise Clarification('Please specify a known place, aerosol, and dates, or configure the fallback model.')
        raw = model_call(model_messages(question, reference_date, user_location))
        result = Interpretation.model_validate_json(raw)
        routing = 'model'
    if result.clarification:
        raise Clarification(result.clarification)
    if result.relationship == 'comparison':
        raise Clarification('Choose one plume; comparison views are not supported yet.')
    if result.variable not in MERRA2_COLUMN_VARIABLES:
        raise Clarification('Choose a column aerosol such as dust, smoke, sulfate, or total aerosols.')
    event = next((e for e in events() if e['id'] == result.event_id), None)
    if result.event_id and event is None:
        raise Clarification('Unknown event. Please provide a location and explicit dates.')
    if event:
        bbox, name = event['bbox'], event['name']
        var = next(p['variable'] for p in event['suggested_plans'] if p['level'] == 'column')
    else:
        if not result.places or (len(result.places) > 1 and result.relationship != 'transport'):
            raise Clarification('Please specify one place or a transport path.')
        if location_bbox:
            name, bbox = 'Your area', location_bbox
        elif result.places == ['Global']:
            name, bbox = 'Global', [-180, -90, 180, 90]
        else:
            name, bbox = geocode.resolve_place(' / '.join(result.places))
        var = result.variable
    explicit = explicit_window(question)
    if not explicit:
        years = re.findall(r'\b(?:19|20)\d{2}\b', question)
        if event:
            requested_year = (datetime.fromisoformat(reference_date).year - 1
                              if re.search(r'\blast year\b', question.lower()) else None)
            if any(y != event['time_start'][:4] for y in years) or (requested_year is not None and requested_year != int(event['time_start'][:4])):
                raise Clarification('The event dates conflict with the requested year. Please specify the intended event or dates.')
        elif result.time_mode == 'latest' and (years or re.search(r'\b(last year|last month|summer|winter|spring|autumn)\b', question.lower())):
            raise Clarification('Please specify the historical dates; a historical request cannot use latest data.')
        elif result.time_mode == 'historical' and not years and re.search(r'\b(that one|years ago|a few years)\b', question.lower()):
            raise Clarification('Please specify the year of that event.')
    notice = None
    requested_start = requested_end = None
    if explicit or event or result.time_mode == 'historical':
        if explicit:
            start, end = explicit
        elif result.start and result.end:
            start = hourly.grid._parse_time(result.start, 'start')
            end = hourly.grid._parse_end_time(result.end, 'end')
        elif event:
            start = hourly.grid._parse_time(event['time_start'], 'start')
            end = start + timedelta(hours=48) - timedelta(seconds=1)
            notice = 'Showing the first 48 catalog hours of this event, not a curated peak window.'
        else:
            raise Clarification('Please supply historical start and end dates.')
        requested_start, requested_end = start.isoformat(), end.isoformat()
        if end < start:
            raise Clarification('The end date must follow the start date.')
        if end - start > timedelta(hours=168):
            end = start + timedelta(hours=48) - timedelta(seconds=1)
            notice = 'Showing the first 48 hours of the requested range. Specify another date range to explore further.'
        if end.date() > datetime.fromisoformat(reference_date).date():
            raise Clarification('MERRA-2 does not provide forecasts. Choose historical dates.')
    else:
        start, end = hourly.latest_window(var, result.duration_hours)
        notice = f'Latest available {result.duration_hours} hours of MERRA-2; this is historical reanalysis, not current conditions.'
    plan = QueryPlan(intent='plume', level='column', source='merra2', variable=var,
                     bbox=bbox, place_name=name, time_start=start, time_end=end,
                     aggregation='hourly', style=StyleSpec(colormap='aod_sequential'),
                     caption=f'{name}: {start:%Y-%m-%d %H:%M} – {end:%Y-%m-%d %H:%M} UTC. ' + (notice or ''))
    return plan, {'routing': routing, 'time_mode': 'historical' if requested_start else 'latest',
                  'requested_start': requested_start, 'requested_end': requested_end,
                  'served_start': start.isoformat(), 'served_end': end.isoformat(), 'notice': notice}
