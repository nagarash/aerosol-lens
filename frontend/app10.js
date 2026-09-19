/* Aerosol Lens frontend: aerosols & particulates only.
 *
 * Data flow:
 *   1. Chat input -> POST {BACKEND_URL}/ask -> {plan, data_url, legend, cached}
 *   2. plan's /grid URL -> GET -> 2D speciated aerosol field (MERRA-2).
 *   3. The field is rendered client-side through a colormap onto an
 *      offscreen canvas and added as a MapLibre `image` source.
 *
 * The live air-quality layer is gone: the Google Air Quality heatmap, its
 * bring-your-own-key settings, and the Air quality / Plume tab switcher
 * were removed. Single aerosol/particulates view.
 *
 * Basemaps: keyless Esri raster styles plus keyless OpenFreeMap vector
 * styles (Positron, Bright). Vector switches replace the whole style, so
 * the data layer specs are remembered and re-added on style load.
 *
 * Multi-day plans play as a video by default: daily /grid frames are
 * fetched (3 at a time, 429-aware), rendered with one shared color
 * scale, smoothed (2x upscale + slight blur so the plume looks organic
 * instead of blocky), and looped. The archive is sparse, so days without
 * coverage are skipped rather than killing the animation. The timeline
 * scrubs frames; the play button pauses/resumes.
 */

const BACKEND_URL = (
  (window.AEROSOL_LENS_CONFIG && window.AEROSOL_LENS_CONFIG.BACKEND_URL) ||
  "http://localhost:8000"
).replace(/\/+$/, "");

// Basemap choices. Raster entries are keyless Esri (tile order z/y/x);
// `style` entries are keyless OpenFreeMap vector styles (MapLibre-native,
// no usage caps). Satellite stays the default: maximum land detail, and
// yellow-orange-red plumes pop on it.
const BASEMAPS = {
  satellite: {
    label: "Satellite",
    tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
    // Reference labels (place names + borders) drawn above imagery.
    overlay: ["https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"],
  },
  streets: {
    label: "Streets",
    tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"],
  },
  natgeo: {
    label: "NatGeo",
    tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}"],
  },
  ocean: {
    label: "Ocean",
    tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}"],
    // Reference labels (place names + boundaries) drawn above the ocean base.
    overlay: ["https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Reference/MapServer/tile/{z}/{y}/{x}"],
  },
  terrain: {
    label: "Terrain",
    tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}"],
  },
  light: {
    label: "Light",
    tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"],
  },
  dark: {
    label: "Dark",
    tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"],
  },
  positron: {
    label: "Positron",
    style: "https://tiles.openfreemap.org/styles/positron",
  },
  bright: {
    label: "Bright",
    style: "https://tiles.openfreemap.org/styles/bright",
  },
};

// One-time cleanup: drop the retired Google Air Quality API key if an
// older version of the app stored one.
try { localStorage.removeItem("google_aq_key"); } catch (_) {}

// Build the inline raster style shell for an Esri basemap entry. Vector
// entries (OpenFreeMap) are applied with map.setStyle() instead.
function makeRasterStyle(cfg) {
  const sources = {
    basemap: {
      type: "raster",
      tiles: cfg.tiles,
      tileSize: 256,
      attribution: "© Esri & contributors",
    },
  };
  const layers = [
    {
      id: "basemap",
      type: "raster",
      source: "basemap",
      paint: { "raster-opacity": 1 },
    },
  ];
  if (cfg.overlay) {
    sources.basemap_labels = {
      type: "raster",
      tiles: cfg.overlay,
      tileSize: 256,
    };
    layers.push({
      id: "basemap-labels",
      type: "raster",
      source: "basemap_labels",
      paint: { "raster-opacity": 1 },
    });
  }
  return { version: 8, sources, layers };
}

const map = new maplibregl.Map({
  container: "map",
  style: makeRasterStyle(BASEMAPS.satellite),
  center: [0, 20],
  zoom: 2,
});

// The current plume data layer, remembered so basemap switches (which
// replace the whole style for vector basemaps) can restore it on top.
let savedDataLayer = null; // {source: {...}, layer: {...}}

function restoreDataLayer() {
  if (!savedDataLayer || map.getLayer("data-layer")) return;
  map.addSource("data-source", savedDataLayer.source);
  map.addLayer(savedDataLayer.layer);
}

// Switch basemap. Raster entries swap tile URLs on the existing sources;
// vector entries replace the whole style, so the data layer is re-added
// once the new style loads. The plume data layer always ends up on top.
function setBasemap(name) {
  const cfg = BASEMAPS[name];
  if (!cfg) return;
  if (cfg.style) {
    // Vector style (OpenFreeMap).
    map.setStyle(cfg.style);
    map.once("style.load", restoreDataLayer);
  } else if (map.getSource("basemap")) {
    // Raster -> raster: cheap tile swap, no style reload.
    map.getSource("basemap").setTiles(cfg.tiles);
    if (cfg.overlay && map.getSource("basemap_labels")) {
      map.getSource("basemap_labels").setTiles(cfg.overlay);
      map.setLayoutProperty("basemap-labels", "visibility", "visible");
    } else if (map.getLayer("basemap-labels")) {
      map.setLayoutProperty("basemap-labels", "visibility", "none");
    }
  } else {
    // Vector -> raster: rebuild the raster shell, then restore data.
    map.setStyle(makeRasterStyle(cfg));
    map.once("style.load", restoreDataLayer);
  }
  try { localStorage.setItem("aerosol_basemap", name); } catch (_) {}
  const sel = document.getElementById("basemap-select");
  if (sel) sel.value = name;
}

map.addControl(new maplibregl.NavigationControl(), "top-right");

// ---------------------------------------------------------------------------
// Small UI helpers

const $ = (id) => document.getElementById(id);

function log(msg) {
  const div = document.createElement("div");
  div.textContent = msg;
  const logEl = $("chat-log");
  logEl.appendChild(div);
  logEl.scrollTop = logEl.scrollHeight;
}

function showLoading(on, label) {
  const el = $("loading");
  el.classList.toggle("hidden", !on);
  if (label) $("loading-label").textContent = label;
}

function showError(message) {
  const el = $("error-banner");
  el.textContent = message;
  el.classList.remove("hidden");
}

function hideError() {
  $("error-banner").classList.add("hidden");
}

/** Build a plain-language Error from a failed backend response. */
async function httpError(res, what) {
  let detail = "";
  let retryAfter = "";
  try {
    const body = await res.json();
    detail = body.detail || body.message || "";
  } catch (_) {
    /* non-JSON error body */
  }
  if (res.status === 429) {
    retryAfter = res.headers.get("Retry-After");
    detail =
      detail ||
      `Rate limited${retryAfter ? ` — retry in ${retryAfter}s` : ""}. ` +
        "The backend throttles grid requests to protect its data quota.";
  }
  const msg = detail ? `${what}: ${detail}` : `${what}: HTTP ${res.status}`;
  const err = new Error(res.status === 429 ? `Rate limited. ${detail}` : msg);
  err.status = res.status;
  return err;
}

// ---------------------------------------------------------------------------
// Ask flow

$("chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = $("chat-input");
  const question = input.value.trim();
  if (!question) return;
  input.value = "";
  hideError();
  log(`You: ${question}`);
  showLoading(true, "Asking the model…");
  try {
    const res = await fetch(`${BACKEND_URL}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!res.ok) throw await httpError(res, "Couldn't understand that question");
    const { plan, data_url, legend, cached } = await res.json();

    log(
      `Plan: ${plan.intent} / ${plan.level} / ${plan.variable}${cached ? " (cached)" : ""}`
    );
    $("chat-caption").textContent = plan.caption || "";
    await renderData(plan, data_url, legend);
  } catch (err) {
    // Plain backend errors, never stack traces.
    showError(err.message || "Something went wrong. Please try again.");
    log(`Error: ${err.message}`);
  } finally {
    showLoading(false);
  }
});

// ---------------------------------------------------------------------------
// Rendering: exactly one aerosol data layer at a time.

function clearDataLayer() {
  stopAnimation();
  for (const id of ["data-layer", "data-source"]) {
    if (map.getLayer(id)) map.removeLayer(id);
    if (map.getSource(id)) map.removeSource(id);
  }
  savedDataLayer = null;
}

async function renderData(plan, data_url, legend) {
  clearDataLayer();
  if (data_url.startsWith("/grid")) {
    await renderPlume(plan, data_url);
  } else if (data_url.startsWith("google://")) {
    // The live air-quality view was removed; the backend may still plan
    // surface-level questions. Guide toward aerosol queries instead.
    renderEmptyLegend();
    log("Aerosol Lens now shows aerosols and particulates only — the live air-quality view was removed. Try e.g. “Show the Saharan dust plume”.");
  } else if (data_url.startsWith("cams://")) {
    renderEmptyLegend();
    log("Historical surface particulate data (CAMS) isn't wired up yet — try an aerosol plume question.");
  } else {
    showError(`No renderer for this data source (${plan.source}).`);
  }
}

function renderEmptyLegend() {
  $("legend-title").textContent = "Aerosols";
  $("legend-bar").style.background = "#333";
  $("legend-min").textContent = "";
  $("legend-mid").textContent = "no data";
  $("legend-max").textContent = "";
  $("legend-meta").textContent = "";
  $("legend-guideline").textContent = "";
}

// --- Aerosols: /grid JSON -> colormap -> canvas -> MapLibre image source. -
// --- Animation state: daily frames that play as a video by default. -----

let animFrames = []; // [{url, date}]
let animIndex = 0;
let animTimer = null;
let animCoords = null;

function stopAnimation() {
  if (animTimer) {
    clearInterval(animTimer);
    animTimer = null;
  }
  animFrames = [];
  animIndex = 0;
  animCoords = null;
  const btn = document.getElementById("play-btn");
  if (btn) btn.textContent = "▶";
}

function isPlaying() {
  return animTimer !== null;
}

function pauseAnimation() {
  if (animTimer) {
    clearInterval(animTimer);
    animTimer = null;
  }
  const btn = document.getElementById("play-btn");
  if (btn) btn.textContent = "▶";
}

function playAnimation() {
  if (animFrames.length < 2 || animTimer) return;
  const btn = document.getElementById("play-btn");
  if (btn) btn.textContent = "⏸";
  animTimer = setInterval(() => {
    showFrame(animIndex + 1 >= animFrames.length ? 0 : animIndex + 1);
  }, 650);
}

/** Days (YYYY-MM-DD) covering the plan window, sampled to at most 31. */
function dayList(t0, t1) {
  const start = new Date(String(t0).slice(0, 10) + "T00:00:00Z");
  const end = new Date(String(t1).slice(0, 10) + "T00:00:00Z");
  if (isNaN(start.getTime()) || isNaN(end.getTime()) || end < start) return [];
  const days = [];
  for (let d = new Date(start); d <= end; d.setUTCDate(d.getUTCDate() + 1)) {
    days.push(d.toISOString().slice(0, 10));
  }
  const MAX = 31;
  if (days.length <= MAX) return days;
  // Sample evenly so long ranges still animate within the frame budget.
  const step = days.length / MAX;
  const sampled = [];
  for (let i = 0; i < MAX; i++) sampled.push(days[Math.floor(i * step)]);
  return sampled;
}

/** Per-day /grid URL derived from the plan's data_url (daily frames). */
function frameUrlFor(data_url, day) {
  const u = new URL(data_url, "http://localhost");
  u.searchParams.set("t0", `${day}T00:00:00+00:00`);
  // t1 must cover the whole day: MERRA-2's hourly steps sit at 00:30..23:30,
  // so a midnight-exact end selects an empty window (HTTP 422).
  u.searchParams.set("t1", `${day}T23:59:59+00:00`);
  u.searchParams.set("agg", "daily");
  return u.pathname + u.search;
}

/** fetch() that honors the backend's 429 + Retry-After instead of dying. */
async function fetchWithRetry(url, label) {
  for (let attempt = 0; attempt < 4; attempt++) {
    const res = await fetch(`${BACKEND_URL}${url}`);
    if (res.ok) return res.json();
    if (res.status === 429) {
      const waitMs =
        Math.max(1, parseInt(res.headers.get("Retry-After") || "5", 10)) * 1000;
      if (label) showLoading(true, `${label} — rate limited, retrying…`);
      await new Promise((r) => setTimeout(r, waitMs));
      continue;
    }
    throw await httpError(res, "Grid request failed");
  }
  throw new Error("Still rate limited after retries — try again in a minute.");
}

/** Fetch frame URLs with limited concurrency; onProgress(done, total).
 * A day that fails terminally (e.g. 422 = no archive coverage) resolves
 * to null so one gap doesn't kill the whole animation. 429s are still
 * retried inside fetchWithRetry. */
async function fetchFrames(urls, onProgress) {
  const results = new Array(urls.length);
  let next = 0;
  let done = 0;
  async function worker() {
    while (next < urls.length) {
      const i = next++;
      try {
        results[i] = await fetchWithRetry(urls[i], `Loading day ${i + 1}/${urls.length}`);
      } catch (_) {
        results[i] = null; // sparse archive day: skipped, not fatal
      }
      done++;
      onProgress(done, urls.length);
    }
  }
  const CONCURRENCY = 3;
  await Promise.all(
    Array.from({ length: Math.min(CONCURRENCY, urls.length) }, worker)
  );
  return results;
}

/** Shared p2/p98 range across frames so colors stay comparable over time. */
function globalRange(grids) {
  return GridRender.normalizeRange(grids.flatMap((g) => g.values));
}

/**
 * 2x smooth upscale + slight blur: kills the blocky "straight grid" look
 * so the plume reads as an organic cloud instead of pasted rectangles.
 */
function smoothCanvas(src) {
  const scale = 2;
  const out = document.createElement("canvas");
  out.width = src.width * scale;
  out.height = src.height * scale;
  const ctx = out.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  try {
    ctx.filter = "blur(1.2px)";
  } catch (_) {
    /* older canvas: smoothing alone still helps */
  }
  ctx.drawImage(src, 0, 0, out.width, out.height);
  try {
    ctx.filter = "none";
  } catch (_) {}
  return out;
}

/** Grid -> smoothed canvas + pixel buffer (buffer carries legend facts). */
function frameCanvas(grid, range) {
  const buf = GridRender.gridToPixelBuffer(grid, range);
  const c = document.createElement("canvas");
  c.width = buf.width;
  c.height = buf.height;
  c.getContext("2d").putImageData(new ImageData(buf.data, buf.width, buf.height), 0, 0);
  return { canvas: smoothCanvas(c), buf };
}

function showFrame(i) {
  if (!animFrames.length) return;
  animIndex = ((i % animFrames.length) + animFrames.length) % animFrames.length;
  const f = animFrames[animIndex];
  const src = map.getSource("data-source");
  if (src) {
    if (typeof src.updateImage === "function") {
      src.updateImage({ url: f.url, coordinates: animCoords });
    } else {
      // Fallback for older MapLibre: re-add the source.
      if (map.getLayer("data-layer")) map.removeLayer("data-layer");
      map.removeSource("data-source");
      map.addSource("data-source", { type: "image", url: f.url, coordinates: animCoords });
      map.addLayer(savedDataLayer.layer);
    }
    savedDataLayer.source.url = f.url;
  }
  // If the source is gone (mid basemap-switch), restoreDataLayer re-adds
  // the current frame once the new style loads.
  const slider = document.getElementById("time-slider");
  if (slider) slider.value = String(animIndex);
  const label = document.getElementById("time-label");
  if (label) label.textContent = f.date;
}

/** Wire the timeline: scrubbing pauses and jumps; button toggles play. */
function setupTimeScrubber() {
  const slider = document.getElementById("time-slider");
  const btn = document.getElementById("play-btn");
  if (slider) {
    slider.min = "0";
    slider.max = String(Math.max(0, animFrames.length - 1));
    slider.value = "0";
    slider.oninput = () => {
      pauseAnimation();
      showFrame(parseInt(slider.value, 10) || 0);
    };
  }
  if (btn) {
    btn.textContent = isPlaying() ? "⏸" : "▶";
    btn.onclick = () => (isPlaying() ? pauseAnimation() : playAnimation());
  }
  const label = document.getElementById("time-label");
  if (label && animFrames.length) {
    label.textContent =
      animFrames.length > 1
        ? `${animFrames[0].date} → ${animFrames[animFrames.length - 1].date}`
        : animFrames[0].date;
  }
}

function addDataLayer(url, coords) {
  // Remember the specs so a basemap switch can re-add the layer on top
  // of the new style (vector basemaps replace the whole style).
  savedDataLayer = {
    source: { type: "image", url, coordinates: coords },
    layer: {
      id: "data-layer",
      type: "raster",
      source: "data-source",
      paint: { "raster-opacity": 0.9, "raster-fade-duration": 0 },
    },
  };
  map.addSource("data-source", savedDataLayer.source);
  map.addLayer(savedDataLayer.layer);
}

async function renderPlume(plan, data_url) {
  const days = dayList(plan.time_start, plan.time_end);
  if (days.length <= 1) {
    // Single day: one smoothed static frame.
    showLoading(true, "Fetching grid data…");
    try {
      const grid = await fetchWithRetry(data_url, "");
      const { canvas, buf } = frameCanvas(grid, {});
      animCoords = GridRender.gridCorners(grid);
      addDataLayer(canvas.toDataURL(), animCoords);
      map.fitBounds(GridRender.gridBounds(grid), { padding: 40 });
      renderPlumeLegend(grid, buf);
      animFrames = [{ url: savedDataLayer.source.url, date: String(plan.time_start).slice(0, 10) }];
      setupTimeScrubber();
    } finally {
      showLoading(false);
    }
    return;
  }
  // Multi-day: fetch daily frames, then autoplay the animation.
  // The archive is sparse — days without coverage are skipped, not fatal.
  const urls = days.map((d) => frameUrlFor(data_url, d));
  showLoading(true, `Loading day 1/${urls.length}…`);
  try {
    const grids = await fetchFrames(urls, (done, total) => {
      showLoading(true, `Loading day ${done}/${total}…`);
    });
    const { kept, skipped } = GridRender.partitionFrames(grids, days);
    if (!kept.length) {
      throw new Error("No daily coverage in this range — try a narrower window.");
    }
    const range = globalRange(kept.map((f) => f.grid));
    animCoords = GridRender.gridCorners(kept[0].grid);
    let firstBuf = null;
    animFrames = kept.map((f, i) => {
      const { canvas, buf } = frameCanvas(f.grid, range);
      if (i === 0) firstBuf = buf;
      return { url: canvas.toDataURL(), date: f.date };
    });
    addDataLayer(animFrames[0].url, animCoords);
    map.fitBounds(GridRender.gridBounds(kept[0].grid), { padding: 40 });
    renderPlumeLegend(kept[0].grid, firstBuf);
    setupTimeScrubber();
    const gapNote = skipped.length
      ? ` (${skipped.length} day${skipped.length > 1 ? "s" : ""} without coverage skipped)`
      : "";
    if (animFrames.length > 1) {
      playAnimation(); // video plays by default
      log(`Playing ${animFrames.length} daily frames${gapNote} — drag the timeline or pause anytime.`);
    } else {
      log(`Only one day with coverage in this range${gapNote}.`);
    }
  } finally {
    showLoading(false);
  }
}

function renderPlumeLegend(grid, buf) {
  // Real colorbar: gradient + min/mid/max from the fetched data, never
  // hardcoded.
  $("legend-title").textContent = GridRender.prettyVariable(grid.variable);
  $("legend-bar").style.background = GridRender.legendGradient(buf.colormap);
  $("legend-min").textContent = GridRender.formatTick(buf.vmin);
  $("legend-mid").textContent = GridRender.formatTick(buf.vmid);
  $("legend-max").textContent = GridRender.formatTick(buf.vmax);
  const t0 = String(grid.time_start || "").slice(0, 10);
  const t1 = String(grid.time_end || "").slice(0, 10);
  $("legend-meta").textContent =
    `${grid.units || ""} · ${t0}${t1 && t1 !== t0 ? " → " + t1 : ""}` +
    `${grid.aggregation ? " · " + grid.aggregation : ""}`;
  $("legend-guideline").textContent = "";
}

// Basemap picker: restore saved choice, wire the select.
try {
  const saved = localStorage.getItem("aerosol_basemap");
  if (saved && BASEMAPS[saved]) {
    // Apply after the style loads so sources exist.
    map.once("load", () => setBasemap(saved));
  }
} catch (_) {}
document.getElementById("basemap-select").addEventListener("change", (e) => {
  setBasemap(e.target.value);
});

// (setupTimeScrubber is defined with the animation code above and wired
// per-render inside renderPlume.)

log("Ask about aerosol plumes and particulates — e.g. “Show me the Saharan dust plume”.");
