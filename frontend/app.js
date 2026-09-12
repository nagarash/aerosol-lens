/* Aerosol Lens frontend.
 *
 * Data flow:
 *   1. Chat input -> POST /ask -> {plan, data_url, legend}
 *   2. data_url "google://..."  -> raster layer calling the Google Air
 *      Quality API heatmap tiles DIRECTLY from the browser with the user's
 *      own key (never proxied through our backend).
 *   3. data_url "/grid?..."     -> fetch grid JSON from our backend and
 *      render client-side (TODO: pick deck.gl heatmap or canvas rendering).
 *   4. Mode toggle switches Air quality (surface) <-> Plume view (column);
 *      the two are NEVER blended in one layer (validator enforces this
 *      server-side too).
 */

const BACKEND = ""; // same origin in docker-compose; override for dev
const GOOGLE_AQ_API_KEY = localStorage.getItem("google_aq_key") || "";
// TODO(ux): add a settings affordance to store the user's Google API key
// (bring-your-own-key). Without it, live/forecast views are disabled.

const map = new maplibregl.Map({
  container: "map",
  style: {
    version: 8,
    sources: {
      // Dark, muted basemap so the data is the hero. Free CARTO tiles.
      basemap: {
        type: "raster",
        tiles: ["https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png"],
        tileSize: 256,
        attribution: "© OpenStreetMap contributors © CARTO",
      },
    },
    layers: [{ id: "basemap", type: "raster", source: "basemap", paint: { "raster-opacity": 0.9 } }],
  },
  center: [0, 20],
  zoom: 2,
});

map.addControl(new maplibregl.NavigationControl(), "top-right");

let currentMode = "health"; // 'health' (surface) | 'plume' (column)

document.querySelectorAll("#mode-toggle button").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#mode-toggle button").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    currentMode = btn.dataset.mode;
    log(`Mode: ${currentMode === "health" ? "Air quality (surface)" : "Plume view (column)"}. Ask a question to load data.`);
    // TODO: re-issue the last question with the flipped intent when a plan exists.
  });
});

document.getElementById("chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = document.getElementById("chat-input");
  const question = input.value.trim();
  if (!question) return;
  input.value = "";
  log(`You: ${question}`);
  log("Thinking…");
  try {
    const res = await fetch(`${BACKEND}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      log(`Error: ${err.detail || res.statusText}`);
      return;
    }
    const { plan, data_url, legend, cached } = await res.json();
    log(`Plan: ${plan.intent} / ${plan.level} / ${plan.variable} ${cached ? "(cached)" : ""}`);
    document.getElementById("chat-caption").textContent = plan.caption || "";
    renderLegend(plan, legend);
    await renderData(plan, data_url);
    map.fitBounds(
      [[plan.bbox[0], plan.bbox[1]], [plan.bbox[2], plan.bbox[3]]],
      { padding: 40 }
    );
    setupTimeScrubber(plan);
  } catch (err) {
    log(`Error: ${err.message}`);
  }
});

async function renderData(plan, data_url) {
  // Remove the previous overlay; one view at a time, never blended.
  for (const id of ["aerosol-layer", "aerosol-source"]) {
    if (map.getLayer(id)) map.removeLayer(id);
    if (map.getSource(id)) map.removeSource(id);
  }

  if (data_url.startsWith("google://")) {
    if (!GOOGLE_AQ_API_KEY) {
      log("Live view needs a Google Air Quality API key (bring-your-own-key). TODO: settings UI.");
      return;
    }
    // Google heatmap tiles, called directly from the browser.
    const template = data_url.replace("google://", "https://airquality.googleapis.com/v1/");
    map.addSource("aerosol-source", {
      type: "raster",
      tiles: [`${template}?key=${GOOGLE_AQ_API_KEY}`],
      tileSize: 256,
    });
    map.addLayer({
      id: "aerosol-layer",
      type: "raster",
      source: "aerosol-source",
      paint: { "raster-opacity": plan.style.opacity ?? 0.65 },
    });
  } else if (data_url.startsWith("/grid")) {
    // TODO(rendering): fetch the grid JSON and render client-side
    // (deck.gl HeatmapLayer or a canvas/WebGL shader with the colormap
    // applied in-shader for instant restyling).
    log(`Grid rendering not yet implemented for ${plan.variable} (${plan.source}).`);
  } else {
    log(`No renderer for data_url: ${data_url}`);
  }
}

function renderLegend(plan, legend) {
  // TODO: build a real colorbar from data/colormaps.json stops keyed by
  // legend.stops, with labeled thresholds and the WHO guideline line for
  // exceedance mode.
  document.getElementById("legend-title").textContent = plan.variable;
  document.getElementById("legend-units").textContent = legend.units || "";
  document.getElementById("legend-guideline").textContent =
    plan.style.mode === "exceedance" ? `Guideline: ${legend.guideline || "WHO 24h"}` : "";
  const bar = document.getElementById("legend-bar");
  bar.style.background =
    plan.level === "surface"
      ? "linear-gradient(to right,#00e400,#ffff00,#ff7e00,#ff0000,#8f3f97,#7e0023)"
      : "linear-gradient(to right,#ffffcc,#fed976,#fd8d3c,#e31a1c,#800026)";
}

function setupTimeScrubber(plan) {
  // TODO(animation): expand plan.time_start..time_end into frames at
  // plan.aggregation steps, prefetch the next N frames while the current
  // one displays, crossfade on advance.
  const label = document.getElementById("time-label");
  label.textContent = `${plan.time_start.slice(0, 10)} → ${plan.time_end.slice(0, 10)}`;
  document.getElementById("play-btn").onclick = () => log("Animation not yet implemented.");
}

function log(msg) {
  const el = document.getElementById("chat-log");
  const div = document.createElement("div");
  div.textContent = msg;
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
}
