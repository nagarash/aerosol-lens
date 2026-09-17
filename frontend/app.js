/* Aerosol Lens frontend.
 *
 * Data flow:
 *   1. Chat input -> POST {BACKEND_URL}/ask -> {plan, data_url, legend, cached}
 *   2. plan.level === 'surface' -> Air quality mode: Google Air Quality API
 *      heatmap tiles called DIRECTLY from the browser with the user's own
 *      key (bring-your-own-key; the backend never sees it).
 *   3. plan.level === 'column'  -> Plume mode: fetch the plan's /grid URL,
 *      render the 2D field client-side through a colormap onto an offscreen
 *      canvas, add it as a MapLibre `image` source.
 *   4. The mode toggle follows the plan: if the plan's level disagrees with
 *      the current toggle, the UI auto-switches with a one-line notice.
 *      Surface and column data are NEVER rendered together.
 */

const BACKEND_URL = (
  (window.AEROSOL_LENS_CONFIG && window.AEROSOL_LENS_CONFIG.BACKEND_URL) ||
  "http://localhost:8000"
).replace(/\/+$/, "");

const GOOGLE_KEY_STORAGE = "google_aq_key";
const GOOGLE_MAPTYPE = "UAQI_RED_GREEN"; // Google's red-green US-AQI heatmap

// Basemap choices (all keyless Esri; tile order z/y/x). Satellite is the
// default: maximum land detail, and yellow-orange-red plumes pop on it.
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
  light: {
    label: "Light",
    tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"],
  },
  dark: {
    label: "Dark",
    tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"],
  },
};

const map = new maplibregl.Map({
  container: "map",
  style: {
    version: 8,
    sources: {
      basemap: {
        type: "raster",
        tiles: BASEMAPS.satellite.tiles,
        tileSize: 256,
        attribution: "© Esri & contributors",
      },
      basemap_labels: {
        type: "raster",
        tiles: BASEMAPS.satellite.overlay,
        tileSize: 256,
      },
    },
    layers: [
      {
        id: "basemap",
        type: "raster",
        source: "basemap",
        paint: { "raster-opacity": 1 },
      },
      {
        id: "basemap-labels",
        type: "raster",
        source: "basemap_labels",
        paint: { "raster-opacity": 1 },
      },
    ],
  },
  center: [0, 20],
  zoom: 2,
});

// Switch basemap; the plume/aqi data layers are added above these, so they
// stay on top. Labels overlay only applies to satellite.
function setBasemap(name) {
  const cfg = BASEMAPS[name];
  if (!cfg || !map.getSource("basemap")) return;
  map.getSource("basemap").setTiles(cfg.tiles);
  if (cfg.overlay) {
    map.getSource("basemap_labels").setTiles(cfg.overlay);
    map.setLayoutProperty("basemap-labels", "visibility", "visible");
  } else {
    map.setLayoutProperty("basemap-labels", "visibility", "none");
  }
  try { localStorage.setItem("aerosol_basemap", name); } catch (_) {}
  const sel = document.getElementById("basemap-select");
  if (sel) sel.value = name;
}

map.addControl(new maplibregl.NavigationControl(), "top-right");

let currentMode = "health"; // 'health' (surface) | 'plume' (column)

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
// Mode toggle: follows the plan, never mixes surface + column.

function setMode(mode, reason) {
  currentMode = mode;
  document.querySelectorAll("#mode-toggle button").forEach((b) =>
    b.classList.toggle("active", b.dataset.mode === mode)
  );
  if (reason) log(reason);
}

function modeForPlan(plan) {
  return plan.level === "surface" ? "health" : "plume";
}

document.querySelectorAll("#mode-toggle button").forEach((btn) => {
  btn.addEventListener("click", () => {
    setMode(
      btn.dataset.mode,
      `Mode: ${btn.dataset.mode === "health" ? "Air quality (surface)" : "Plume view (column)"}. Ask a question to load data.`
    );
    clearDataLayer();
  });
});

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

    const wantMode = modeForPlan(plan);
    if (wantMode !== currentMode) {
      setMode(
        wantMode,
        `Switched to ${wantMode === "health" ? "Air quality" : "Plume view"} — this plan is ${plan.level}-level data.`
      );
    }
    log(
      `Plan: ${plan.intent} / ${plan.level} / ${plan.variable}${cached ? " (cached)" : ""}`
    );
    $("chat-caption").textContent = plan.caption || "";
    await renderData(plan, data_url, legend);
    setupTimeScrubber(plan);
  } catch (err) {
    // Plain backend errors, never stack traces.
    showError(err.message || "Something went wrong. Please try again.");
    log(`Error: ${err.message}`);
  } finally {
    showLoading(false);
  }
});

// ---------------------------------------------------------------------------
// Rendering: exactly one data layer at a time.

function clearDataLayer() {
  for (const id of ["data-layer", "data-source", "mask-layer", "mask-source", "bbox-line", "bbox-source"]) {
    if (map.getLayer(id)) map.removeLayer(id);
    if (map.getSource(id)) map.removeSource(id);
  }
}

async function renderData(plan, data_url, legend) {
  clearDataLayer();
  if (data_url.startsWith("google://")) {
    renderAirQuality(plan, data_url);
  } else if (data_url.startsWith("/grid")) {
    await renderPlume(plan, data_url);
  } else if (data_url.startsWith("cams://")) {
    log("Historical surface data (CAMS) isn't wired up yet — try a live air-quality question or a plume query.");
  } else {
    showError(`No renderer for this data source (${plan.source}).`);
  }
}

// --- Air quality: Google heatmap tiles, key stays in the browser. ----------

function googleTileTemplate(data_url) {
  let template = data_url.replace(
    "google://",
    "https://airquality.googleapis.com/v1/"
  );
  if (!template.includes("mapTypes/")) {
    template = template.replace(
      "/heatmapTiles/",
      `/mapTypes/${GOOGLE_MAPTYPE}/heatmapTiles/`
    );
  }
  return template;
}

function renderAirQuality(plan, data_url) {
  const key = localStorage.getItem(GOOGLE_KEY_STORAGE) || "";
  if (!key) {
    // Friendly empty state: no key, no layer, settings opened for them.
    log("Live air-quality view needs a Google Air Quality API key (bring-your-own-key). Add it in Settings — it stays in this browser.");
    openSettings();
    renderSurfaceLegendEmpty();
    return;
  }
  const template = googleTileTemplate(data_url);
  map.addSource("data-source", {
    type: "raster",
    tiles: [`${template}?key=${encodeURIComponent(key)}`],
    tileSize: 256,
    attribution: "Air quality: Google Air Quality API",
  });
  map.addLayer({
    id: "data-layer",
    type: "raster",
    source: "data-source",
    paint: { "raster-opacity": plan.style?.opacity ?? 0.65 },
  });
  map.fitBounds(
    [
      [plan.bbox[0], plan.bbox[1]],
      [plan.bbox[2], plan.bbox[3]],
    ],
    { padding: 40 }
  );
  addSpotlightMask(plan.bbox);
  renderSurfaceLegend(plan);
}

// Google's heatmap tiles are global pre-rendered rasters: they paint every
// region, not just the queried place. Dim everything outside the plan's bbox
// (spotlight mask) and outline the bbox so the queried area reads clearly.
function addSpotlightMask(bbox) {
  const [w, s, e, n] = bbox;
  map.addSource("mask-source", {
    type: "geojson",
    data: {
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          properties: {},
          geometry: {
            type: "Polygon",
            coordinates: [
              [[-180, -90], [180, -90], [180, 90], [-180, 90], [-180, -90]],
              [[w, s], [w, n], [e, n], [e, s], [w, s]],
            ],
          },
        },
      ],
    },
  });
  map.addLayer({
    id: "mask-layer",
    type: "fill",
    source: "mask-source",
    paint: { "fill-color": "#000000", "fill-opacity": 0.5 },
  });
  map.addSource("bbox-source", {
    type: "geojson",
    data: {
      type: "Feature",
      properties: {},
      geometry: {
        type: "LineString",
        coordinates: [[w, s], [w, n], [e, n], [e, s], [w, s]],
      },
    },
  });
  map.addLayer({
    id: "bbox-line",
    type: "line",
    source: "bbox-source",
    paint: { "line-color": "#ffffff", "line-width": 2, "line-opacity": 0.9 },
  });
}

function renderSurfaceLegend(plan) {
  // Google's UAQI_RED_GREEN tiles use the US AQI color scale; the bar is
  // fixed because the tile colors are fixed server-side.
  $("legend-title").textContent = "Air quality (US AQI)";
  $("legend-bar").style.background =
    "linear-gradient(to right,#00e400,#ffff00,#ff7e00,#ff0000,#8f3f97,#7e0023)";
  $("legend-min").textContent = "Good";
  $("legend-mid").textContent = "";
  $("legend-max").textContent = "Hazardous";
  $("legend-meta").textContent = "Live heatmap · Google Air Quality API";
  $("legend-guideline").textContent =
    plan.style?.mode === "exceedance" ? "Guideline: WHO 24h PM2.5" : "";
}

function renderSurfaceLegendEmpty() {
  $("legend-title").textContent = "Air quality";
  $("legend-bar").style.background = "#333";
  $("legend-min").textContent = "";
  $("legend-mid").textContent = "no API key";
  $("legend-max").textContent = "";
  $("legend-meta").textContent = "Add a key in Settings (⚙)";
  $("legend-guideline").textContent = "";
}

// --- Plume: /grid JSON -> colormap -> canvas -> MapLibre image source. -----

async function renderPlume(plan, data_url) {
  showLoading(true, "Fetching grid data…");
  try {
    const res = await fetch(`${BACKEND_URL}${data_url}`);
    if (!res.ok) throw await httpError(res, "Grid request failed");
    const grid = await res.json();
    const buf = GridRender.gridToPixelBuffer(grid);

    const canvas = document.createElement("canvas");
    canvas.width = buf.width;
    canvas.height = buf.height;
    const ctx = canvas.getContext("2d");
    ctx.putImageData(new ImageData(buf.data, buf.width, buf.height), 0, 0);

    map.addSource("data-source", {
      type: "image",
      url: canvas.toDataURL(),
      coordinates: GridRender.gridCorners(grid),
    });
    map.addLayer({
      id: "data-layer",
      type: "raster",
      source: "data-source",
      paint: { "raster-opacity": 0.9, "raster-fade-duration": 0 },
    });
    map.fitBounds(GridRender.gridBounds(grid), { padding: 40 });
    renderPlumeLegend(grid, buf);
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

// ---------------------------------------------------------------------------
// Settings: Google API key (bring-your-own-key, localStorage only).

function openSettings() {
  $("settings-key").value = localStorage.getItem(GOOGLE_KEY_STORAGE) || "";
  $("settings").classList.remove("hidden");
}

function closeSettings() {
  $("settings").classList.add("hidden");
}

$("settings-btn").addEventListener("click", () => {
  $("settings").classList.toggle("hidden");
  if (!$("settings").classList.contains("hidden")) openSettings();
});
$("settings-close").addEventListener("click", closeSettings);
$("settings-save").addEventListener("click", () => {
  const key = $("settings-key").value.trim();
  if (key) {
    localStorage.setItem(GOOGLE_KEY_STORAGE, key);
    log("Google API key saved — it stays in this browser and is only sent to Google.");
  } else {
    localStorage.removeItem(GOOGLE_KEY_STORAGE);
    log("Google API key removed.");
  }
  closeSettings();
});
$("settings-clear").addEventListener("click", () => {
  localStorage.removeItem(GOOGLE_KEY_STORAGE);
  $("settings-key").value = "";
  log("Google API key removed.");
});

// ---------------------------------------------------------------------------
// Time scrubber (stub: per-frame prefetching is a later increment).

function setupTimeScrubber(plan) {
  $("time-label").textContent = `${String(plan.time_start).slice(0, 10)} → ${String(plan.time_end).slice(0, 10)}`;
  $("play-btn").onclick = () => log("Animation not yet implemented.");
}

log("Ask about air quality or aerosol plumes — e.g. “Is it safe to run in Delhi today?”");
