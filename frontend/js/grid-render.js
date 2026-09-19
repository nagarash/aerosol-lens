/* Aerosol Lens: DOM-free grid -> pixel-buffer rendering logic.
 *
 * Pure functions operating on plain arrays. No DOM, no canvas, no
 * MapLibre: given a /grid response object, returns a ready-to-paint
 * RGBA buffer plus the legend facts (min/mid/max, colormap). The app
 * paints the buffer onto an offscreen canvas and hands the data URL to
 * MapLibre as an `image` source.
 *
 * Grid contract (backend/grid.py -> GET /grid):
 *   { variable, units, lats[], lons[], values[][], nx, ny,
 *     time_start, time_end, time_steps_used, aggregation,
 *     downsampled, source }
 * - lats: ASCENDING (south -> north).
 * - lons: slice order (ascending, except antimeridian-crossing bboxes,
 *   where they wrap, e.g. [170, 175, -175, -170]).
 * - values[r][c]: r indexes lats (ascending), c indexes lons.
 * - values rounded to 4 decimals; missing cells are null/NaN.
 *
 * Normalization: data-driven p2/p98 percentiles, not absolute min/max.
 * AOD retrievals have hot-pixel outliers; normalizing to the raw max
 * would crush the whole plume into the bottom of the scale. p2/p98
 * keeps the color ramp honest about the bulk of the data. Constant
 * grids (vmin == vmax) are expanded slightly to avoid div-by-zero.
 */

const COLORMAPS = {
  // Yellow -> orange -> red sequential, for dust AOD (DU* variables).
  dust: [
    [0.0, "#ffffcc"],
    [0.125, "#ffeda0"],
    [0.25, "#fed976"],
    [0.375, "#feb24c"],
    [0.5, "#fd8d3c"],
    [0.625, "#fc4e2a"],
    [0.75, "#e31a1c"],
    [0.875, "#bd0026"],
    [1.0, "#800026"],
  ],
  // Viridis-like perceptually-uniform default for all other variables.
  viridis: [
    [0.0, "#440154"],
    [0.125, "#482878"],
    [0.25, "#3f4788"],
    [0.375, "#31688e"],
    [0.5, "#287d8e"],
    [0.625, "#1f968b"],
    [0.75, "#29af7f"],
    [0.875, "#5ec962"],
    [1.0, "#fde725"],
  ],
};

const PRETTY_VARIABLES = {
  DUEXTTAU: "Dust AOD",
  TOTEXTTAU: "Total AOD",
  BCEXTTAU: "Black carbon AOD",
  OCEXTTAU: "Organic carbon AOD",
  SUEXTTAU: "Sulfate AOD",
  SSEXTTAU: "Sea salt AOD",
};

function hexToRgb(hex) {
  const m = /^#([0-9a-f]{6})$/i.exec(hex);
  if (!m) throw new Error(`bad hex color ${hex}`);
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** Sample a colormap stop-list at t in [0,1] -> [r, g, b]. */
function sampleColormap(stops, t) {
  const x = Math.min(1, Math.max(0, t));
  for (let i = 1; i < stops.length; i++) {
    if (x <= stops[i][0]) {
      const [t0, c0] = stops[i - 1];
      const [t1, c1] = stops[i];
      const f = t1 === t0 ? 0 : (x - t0) / (t1 - t0);
      const [r0, g0, b0] = hexToRgb(c0);
      const [r1, g1, b1] = hexToRgb(c1);
      return [
        Math.round(r0 + (r1 - r0) * f),
        Math.round(g0 + (g1 - g0) * f),
        Math.round(b0 + (b1 - b0) * f),
      ];
    }
  }
  return hexToRgb(stops[stops.length - 1][1]);
}

function colormapForVariable(variable) {
  // Dust aerosol optical depth gets the yellow-orange-red ramp;
  // everything else uses the viridis-like default.
  return String(variable || "").toUpperCase().includes("DU")
    ? "dust"
    : "viridis";
}

function prettyVariable(variable) {
  return PRETTY_VARIABLES[String(variable || "").toUpperCase()] || String(variable);
}

function percentile(sorted, q) {
  if (sorted.length === 0) return NaN;
  if (sorted.length === 1) return sorted[0];
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/** Smoothstep in [0,1]: 0 at/below a, 1 at/above b, smooth between. */
function smoothstep(a, b, x) {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

/** Data-driven range: p2/p98 over finite values. Returns {vmin, vmax, vmid}. */
function normalizeRange(values2d) {
  const flat = [];
  for (const row of values2d) {
    for (const v of row) {
      const n = Number(v);
      if (Number.isFinite(n)) flat.push(n);
    }
  }
  if (flat.length === 0) return { vmin: 0, vmax: 1, vmid: 0.5, empty: true };
  flat.sort((a, b) => a - b);
  let vmin = percentile(flat, 0.02);
  let vmax = percentile(flat, 0.98);
  if (!(vmax > vmin)) {
    // Constant (or single-valued) grid: expand so the ramp is defined.
    const c = vmin;
    const eps = Math.abs(c) > 0 ? Math.abs(c) * 0.01 : 1e-6;
    vmin = c - eps;
    vmax = c + eps;
  }
  return { vmin, vmax, vmid: (vmin + vmax) / 2, empty: false };
}

/**
 * Unwrap longitudes for antimeridian-crossing grids so coordinates are
 * continuous (e.g. [170, 175, -175, -170] -> [170, 175, 185, 190]).
 * MapLibre renders lng > 180 fine; the wrap would otherwise tear the image.
 */
function unwrapLons(lons) {
  const out = [];
  let offset = 0;
  for (let i = 0; i < lons.length; i++) {
    if (i > 0 && lons[i] < lons[i - 1]) offset += 360;
    out.push(lons[i] + offset);
  }
  return out;
}

/** MapLibre `image` source corners: [top-left, top-right, bottom-right, bottom-left]. */
function gridCorners(grid) {
  const lons = unwrapLons(grid.lons);
  const lats = grid.lats;
  const left = lons[0];
  const right = lons[lons.length - 1];
  const bottom = lats[0];
  const top = lats[lats.length - 1];
  return [
    [left, top],
    [right, top],
    [right, bottom],
    [left, bottom],
  ];
}

/** [[w, s], [e, n]] bounds for map.fitBounds (lons unwrapped). */
function gridBounds(grid) {
  const lons = unwrapLons(grid.lons);
  const lats = grid.lats;
  return [
    [lons[0], lats[0]],
    [lons[lons.length - 1], lats[lats.length - 1]],
  ];
}

/**
 * Render the grid to an RGBA pixel buffer.
 * Returns {width, height, data (Uint8ClampedArray), vmin, vmid, vmax,
 * colormap}. Canvas row 0 (top) is the northernmost latitude; NaN/null
 * cells are transparent. Low values fade to transparent (only the plume
 * body is painted) and the bbox edges are feathered, so the overlay melts
 * into the basemap instead of pasting a hard rectangle.
 */
function gridToPixelBuffer(grid, opts = {}) {
  const { lats, lons, values } = grid;
  if (!Array.isArray(lats) || !Array.isArray(lons) || !Array.isArray(values)) {
    throw new Error("grid needs lats[], lons[], values[][] arrays");
  }
  const ny = lats.length;
  const nx = lons.length;
  if (values.length !== ny) {
    throw new Error(`values has ${values.length} rows but lats has ${ny}`);
  }
  for (let r = 0; r < ny; r++) {
    if (!Array.isArray(values[r]) || values[r].length !== nx) {
      throw new Error(`values row ${r} has ${values[r] && values[r].length} cols, expected ${nx}`);
    }
  }
  const name = opts.colormap || colormapForVariable(grid.variable);
  const stops = COLORMAPS[name];
  if (!stops) throw new Error(`unknown colormap ${name}`);
  // Fixed range override (e.g. shared across animation frames so colors
  // stay comparable over time); otherwise data-driven p2/p98.
  const fixed =
    Number.isFinite(opts.vmin) &&
    Number.isFinite(opts.vmax) &&
    opts.vmax > opts.vmin;
  const { vmin, vmax, vmid } = fixed
    ? {
        vmin: opts.vmin,
        vmax: opts.vmax,
        vmid: Number.isFinite(opts.vmid) ? opts.vmid : (opts.vmin + opts.vmax) / 2,
      }
    : normalizeRange(values);
  const span = vmax - vmin;
  const data = new Uint8ClampedArray(nx * ny * 4);
  // Feather the bbox border (in grid pixels) so no hard rectangle edge
  // shows on the map. Skipped on tiny grids where every pixel is a border.
  const featherPx =
    Math.min(nx, ny) >= 6 ? Math.max(2, Math.round(Math.min(nx, ny) * 0.05)) : 0;
  for (let r = 0; r < ny; r++) {
    // lats ascend south->north; canvas row 0 is the top (north).
    const outRow = ny - 1 - r;
    for (let c = 0; c < nx; c++) {
      // null/undefined are missing cells (Number(null) === 0 would lie).
      const raw = values[r][c];
      const v = raw === null || raw === undefined ? NaN : Number(raw);
      const i = (outRow * nx + c) * 4;
      if (!Number.isFinite(v)) {
        data[i + 3] = 0; // transparent: no data here
        continue;
      }
      const t = Math.min(1, Math.max(0, (v - vmin) / span));
      // Background haze fades out: alpha ramps from 0 at the bottom of the
      // scale to full by t=0.35, so only the plume body gets painted.
      const valueAlpha = smoothstep(0.04, 0.35, t);
      let edgeAlpha = 1;
      if (featherPx > 0) {
        const d = Math.min(c, nx - 1 - c, outRow, ny - 1 - outRow);
        edgeAlpha = Math.min(1, d / featherPx);
      }
      const [rr, gg, bb] = sampleColormap(stops, t);
      data[i] = rr;
      data[i + 1] = gg;
      data[i + 2] = bb;
      data[i + 3] = Math.round(235 * valueAlpha * edgeAlpha);
    }
  }
  return { width: nx, height: ny, data, vmin, vmid, vmax, colormap: name };
}

/** CSS gradient for the colorbar legend, sampled from the colormap. */
function legendGradient(name) {
  const stops = COLORMAPS[name] || COLORMAPS.viridis;
  const parts = stops.map(([t, c]) => `${c} ${(t * 100).toFixed(1)}%`);
  return `linear-gradient(to right, ${parts.join(", ")})`;
}

/** Compact tick formatting: 0.0065 -> "0.0065", 2.11 -> "2.11". */
function formatTick(v) {
  if (!Number.isFinite(v)) return "–";
  return String(Number(v.toPrecision(3)));
}

/**
 * Split fetched daily frames into covered vs missing days.
 * grids[i] is the /grid result for days[i], or null when that day failed
 * terminally (e.g. HTTP 422: no archive coverage). Order is preserved so
 * the animation still plays chronologically over the surviving days.
 */
function partitionFrames(grids, days) {
  const kept = [];
  const skipped = [];
  for (let i = 0; i < days.length; i++) {
    if (grids[i]) kept.push({ grid: grids[i], date: days[i] });
    else skipped.push(days[i]);
  }
  return { kept, skipped };
}

const GridRender = {
  COLORMAPS,
  hexToRgb,
  sampleColormap,
  colormapForVariable,
  prettyVariable,
  percentile,
  smoothstep,
  normalizeRange,
  unwrapLons,
  gridCorners,
  gridBounds,
  gridToPixelBuffer,
  legendGradient,
  formatTick,
  partitionFrames,
};

if (typeof module !== "undefined" && module.exports) {
  module.exports = GridRender;
} else if (typeof window !== "undefined") {
  window.GridRender = GridRender;
}
