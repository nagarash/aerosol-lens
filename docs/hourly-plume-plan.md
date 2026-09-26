# Hourly plume viewer implementation plan

## Product contract

Render MERRA-2 column aerosol optical depth as hourly frames. Default to the latest complete 48 hours for the selected variable. Recent relative requests use published coverage, not today's date minus a constant. Preserve explicit historical dates and named-event dates. Label served dates and any recent-data substitution. No forecasts or breathing-safety answers.

## Architecture

Local extraction runs first (place candidates, aerosol vocabulary, explicit date ranges, recency expressions, event aliases). Fully resolved queries make zero model calls. Unresolved queries make at most one general-model call, followed by deterministic resolution and validation; Jev is removed from the request chain. The model may return names or IDs, aerosol choices, date semantics, or a clarification. It never controls coordinates, archive availability, source, or rendering settings. Do not silently choose one side of a comparison or infer a historical event from vague wording.

## Phase 1: contracts and correctness

- Add hourly-frame contracts independent of aggregate /grid responses; preserve the existing aggregate endpoint.
- Separate requested dates from effective dates and expose routing/time-resolution metadata.
- Support antimeridian boxes throughout validation and viewport union.
- Prevent the daily-mean store from answering sub-day requests.
- Test exact 48-frame selection, timestamp ordering, missing values, historical dates, and wrapped geography.

## Phase 2: hourly storage and retrieval

- Store per-variable/per-day hourly arrays separately from daily means. Atomic publication makes a day visible only after all 24 timestamps and shapes validate.
- Start with a configurable seven-published-day rolling window, bounded historical cache, and pinned representative event windows. No all-history download.
- Determine latest coverage per variable from complete published hours, requiring 48 contiguous hours. Never infer hourly coverage from daily means.
- Ingest from existing authenticated NASA/kerchunk helpers with bounded concurrency. Retain hourly fields. Idempotent ingestion and a CLI support incremental runs.
- Cold historical reads preserve time, crop spatially, fetch independent granules concurrently, and cache reusable hourly data. Distinguish unavailable dates from operational errors.
- Benchmark six-hour spatial chunks against hourly chunks before choosing a production optimum. Bound in-flight memory for the 1 GB deployment.

## Phase 3: deterministic-first routing

- Extract independent place mentions with spans; discard nested matches, preserve genuine separate mentions.
- Prefer US states for bare ambiguous names, but retain qualified alternatives and stable IDs. Curated exceptions cover well-known smaller cities; population is not a fame metric.
- Use explicit aerosol mappings and an explicit default total-AOD policy; ambiguous species trigger fallback.
- Parse explicit date ranges before single dates. Recent requests default to 48 available hours; explicit historical ranges remain unchanged.
- Load the event catalog. Require reviewed representative windows rather than inventing peak-event dates; until reviewed, use the first 48 catalog hours with a clear label. Explicit dates override catalog dates.
- Use one general LLM fallback with a small extraction schema; validate output once, then clarify rather than adding a serial model retry. Keep health refusals and unsupported-comparison responses.

## Phase 4: API and progressive browser playback

- Return a frame manifest with UTC timestamps, coordinates, fixed color scale, served/requested windows, and cacheable six-hour batch URLs.
- Encode batches as little-endian float32 with explicit shape and missing-value semantics; gzip in transit. Version caches by dataset and encoding.
- Start display after the first batch. Prefetch with bounded concurrency, cancel obsolete requests, and reuse one canvas/image source. Do not wait for all 48 frames or create base64 images for every frame.
- Label hourly timestamps and actual data dates; preserve gaps instead of interpolating them as observations.
- Keep aggregate /grid rendering available for existing clients.

## Phase 5: evals, operations, and rollout

- Replace permissive model-only checks with end-to-end fixed-coverage cases. Include recent substitution, historical events, explicit dates, state/country overrides, transport, comparisons, health, no-location, and malformed model output.
- Assert model-call counts: zero for resolved questions, at most one otherwise. No production API keys needed for offline tests.
- Measure routing, storage read, serialization, first visible frame, buffered playback, and p50/p95 separately for warm and cold paths. No latency claims without measurements.
- Document ingestion, retention, cache limits, deployment volume needs, and optional model configuration. Enable hourly routing only once the store and frontend agree on the contract.
- Deploy only after approval; do not start paid model evals or large NASA backfills merely to run unit tests.

## Acceptance criteria

A common recent query renders 48 distinct hourly frames with zero model calls and zero NASA calls when warm. Historical queries retain their dates. Ambiguous queries use at most one model call. No daily mean can masquerade as an hourly frame. Partial writes never appear in coverage. Browser playback starts before all frames download. Existing aggregate callers continue to work.

## Implementation status

Implemented as an opt-in hourly path: validated atomic hourly storage; bounded
cold-day reuse; retention and event pinning; contiguous per-variable coverage;
local-first routing with one general-model fallback; event grounding; explicit
state/country override; antimeridian validation/union; frame manifest and binary
batches; progressive canvas playback; offline acceptance tests and a live-model
eval option. Existing daily endpoints and their tests are retained.

Initial rollout choices: six-hour chunks and sequential cold variable-day reads
bound memory; no model or NASA performance numbers are claimed. Peak-event
windows remain unreviewed, so initial event playback explicitly uses the first
48 catalog hours. The model remains configurable; no winner is chosen without
live evaluation. The existing gazetteer still uses approximate region/city
viewports and is not a full hierarchical geocoder.

Operational work before production enablement: refresh the index, ingest real
hourly coverage, run live model evals, measure cold retrieval and first-frame
latency on the deployment, and then set HOURLY_PLUMES_ENABLED=1. An external
scheduler should run indexing/ingestion; no deployment or recurring job has
been started by this implementation.
