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
 */

const BACKEND_URL = (
  (window.AEROSOL_LENS_CONFIG && window.AEROSOL_LENS_CONFIG.BACKEND_URL) ||
  "http://localhost:8000"
).replace(/\/+$/, "");

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

// One-time cleanup: drop the retired Google Air Quality API key if an
// older version of the app stored one.
try { localStorage.removeItem("google_aq_key"); } catch (_) {}

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

// Switch basemap; the plume data layer is added above these, so it stays
// on top. Labels overlay only applies to satellite.
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
// Rendering: exactly one aerosol data layer at a time.

function clearDataLayer() {
  for (const id of ["data-layer", "data-source"]) {
    if (map.getLayer(id)) map.removeLayer(id);
    if (map.getSource(id)) map.removeSource(id);
  }
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
// Time scrubber (stub: per-frame prefetching is a later increment).

function setupTimeScrubber(plan) {
  $("time-label").textContent = `${String(plan.time_start).slice(0, 10)} → ${String(plan.time_end).slice(0, 10)}`;
  $("play-btn").onclick = () => log("Animation not yet implemented.");
}

log("Ask about aerosol plumes and particulates — e.g. “Show me the Saharan dust plume”.");
