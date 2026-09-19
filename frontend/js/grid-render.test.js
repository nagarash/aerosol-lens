/* Node tests for frontend/js/grid-render.js (pure logic, no DOM).
 * Run: node frontend/js/grid-render.test.js
 */
const assert = require("node:assert/strict");
const GR = require("./grid-render.js");

let n = 0;
function check(name, fn) {
  fn();
  n++;
  console.log(`ok ${n} - ${name}`);
}

// 1. Colormap endpoints return the exact stop colors.
check("dust colormap endpoints", () => {
  assert.deepEqual(GR.sampleColormap(GR.COLORMAPS.dust, 0), [255, 247, 243]);
  assert.deepEqual(GR.sampleColormap(GR.COLORMAPS.dust, 1), [73, 0, 106]);
});
check("viridis colormap endpoints", () => {
  assert.deepEqual(GR.sampleColormap(GR.COLORMAPS.viridis, 0), [68, 1, 84]);
  assert.deepEqual(GR.sampleColormap(GR.COLORMAPS.viridis, 1), [253, 231, 37]);
});
check("colormap midpoint interpolates between stops", () => {
  const [r, g, b] = GR.sampleColormap(GR.COLORMAPS.dust, 0.5);
  assert.deepEqual([r, g, b], [247, 104, 161]); // #f768a1 exactly at t=0.5
});
check("colormapForVariable picks dust for DU* and viridis otherwise", () => {
  assert.equal(GR.colormapForVariable("DUEXTTAU"), "dust");
  assert.equal(GR.colormapForVariable("TOTEXTTAU"), "viridis");
  assert.equal(GR.colormapForVariable("BCEXTTAU"), "viridis");
});

// 2. NaN / null / missing -> transparent pixel.
check("NaN values render transparent", () => {
  const grid = {
    variable: "DUEXTTAU",
    lats: [10, 20],
    lons: [0, 10],
    values: [
      [NaN, 1.0],
      [null, 0.5],
    ],
  };
  const buf = GR.gridToPixelBuffer(grid);
  // row r=0 (lat 10, south) -> canvas bottom row (y=1)
  const a00 = buf.data[(1 * 2 + 0) * 4 + 3];
  const a01 = buf.data[(1 * 2 + 1) * 4 + 3];
  const a10 = buf.data[(0 * 2 + 0) * 4 + 3];
  assert.equal(a00, 0, "NaN -> alpha 0");
  assert.equal(a10, 0, "null -> alpha 0");
  assert.ok(a01 > 0, "valid value -> opaque");
});

// 3. Normalization of a known gradient is data-driven (p2/p98).
check("normalization uses p2/p98 of a uniform gradient", () => {
  const vals = [];
  for (let i = 0; i <= 100; i++) vals.push([i / 100]);
  const { vmin, vmax, vmid } = GR.normalizeRange(vals);
  assert.ok(Math.abs(vmin - 0.02) < 1e-9, `vmin≈0.02, got ${vmin}`);
  assert.ok(Math.abs(vmax - 0.98) < 1e-9, `vmax≈0.98, got ${vmax}`);
  assert.ok(Math.abs(vmid - 0.5) < 1e-9, `vmid≈0.5, got ${vmid}`);
});
check("outlier does not crush the ramp (p98, not max)", () => {
  const vals = [];
  for (let i = 0; i < 100; i++) vals.push([i / 100]); // 0..0.99
  vals.push([50.0]); // one hot-pixel spike
  const { vmin, vmax } = GR.normalizeRange(vals);
  assert.ok(vmax < 1.5, `vmax should ignore the 50.0 spike, got ${vmax}`);
  assert.ok(vmin >= 0 && vmin < 0.1, `vmin sane, got ${vmin}`);
});
check("constant grid gets a defined (expanded) range", () => {
  const { vmin, vmax } = GR.normalizeRange([
    [2, 2],
    [2, 2],
  ]);
  assert.ok(vmax > vmin, "range must be non-degenerate");
  const buf = GR.gridToPixelBuffer({
    variable: "TOTEXTTAU",
    lats: [0, 1],
    lons: [0, 1],
    values: [
      [2, 2],
      [2, 2],
    ],
  });
  for (let i = 3; i < buf.data.length; i += 4) {
    assert.ok(buf.data[i] > 0, "constant valid grid -> opaque pixels");
  }
});

// 4. Pixel dimensions follow the grid; canvas row 0 is north.
check("pixel buffer dimensions match grid (width=nx, height=ny)", () => {
  const buf = GR.gridToPixelBuffer({
    variable: "DUEXTTAU",
    lats: [10, 20, 30],
    lons: [0, 10, 20, 30],
    values: [
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
    ],
  });
  assert.equal(buf.width, 4);
  assert.equal(buf.height, 3);
  assert.equal(buf.data.length, 4 * 3 * 4);
});
check("northernmost latitude lands on canvas top row", () => {
  const grid = {
    variable: "DUEXTTAU",
    lats: [10, 20, 30], // ascending: row 0 = south
    lons: [0],
    values: [[0.0], [0.5], [1.0]], // brightest in the north
  };
  const buf = GR.gridToPixelBuffer(grid, { colormap: "dust" });
  const top = [buf.data[0], buf.data[1], buf.data[2]]; // canvas y=0
  const bottom = [buf.data[8], buf.data[9], buf.data[10]]; // canvas y=2
  assert.deepEqual(top, [73, 0, 106], "top row = dust(1.0)");
  assert.deepEqual(bottom, [255, 247, 243], "bottom row = dust(0.0)");
});
check("shape mismatch throws an honest error", () => {
  assert.throws(
    () =>
      GR.gridToPixelBuffer({
        variable: "DUEXTTAU",
        lats: [10, 20],
        lons: [0, 10],
        values: [[1, 2]], // only 1 row
      }),
    /rows/
  );
});

// 5. Antimeridian lon unwrap + corners/bounds.
check("unwrapLons makes wrapped longitudes continuous", () => {
  assert.deepEqual(GR.unwrapLons([170, 175, -175, -170]), [170, 175, 185, 190]);
  assert.deepEqual(GR.unwrapLons([-10, 0, 10]), [-10, 0, 10]);
});
check("gridCorners returns MapLibre image-source corner order", () => {
  const corners = GR.gridCorners({
    lats: [10, 20],
    lons: [170, 175, -175, -170],
    values: [],
  });
  assert.deepEqual(corners, [
    [170, 20],
    [190, 20],
    [190, 10],
    [170, 10],
  ]);
});
check("gridBounds unwraps for fitBounds", () => {
  assert.deepEqual(
    GR.gridBounds({ lats: [10, 20], lons: [170, 175, -175, -170], values: [] }),
    [
      [170, 10],
      [190, 20],
    ]
  );
});

// 6. Legend helpers.
check("legendGradient is a CSS gradient over the colormap", () => {
  const g = GR.legendGradient("dust");
  assert.ok(g.startsWith("linear-gradient(to right,"), g);
  assert.ok(g.includes("#fff7f3") && g.includes("#49006a"), g);
});
check("formatTick keeps AOD-scale precision", () => {
  assert.equal(GR.formatTick(0.0065), "0.0065");
  assert.equal(GR.formatTick(2.11), "2.11");
  assert.equal(GR.formatTick(0.336), "0.336");
});
check("prettyVariable names the aerosol", () => {
  assert.equal(GR.prettyVariable("DUEXTTAU"), "Dust AOD");
  assert.equal(GR.prettyVariable("TOTEXTTAU"), "Total AOD");
  assert.equal(GR.prettyVariable("whatever"), "whatever");
});

// 7. Plume fade: low values transparent, edges feathered.
check("smoothstep ramps 0->1 between the stops", () => {
  assert.equal(GR.smoothstep(0.04, 0.35, 0.0), 0);
  assert.equal(GR.smoothstep(0.04, 0.35, 0.04), 0);
  assert.equal(GR.smoothstep(0.04, 0.35, 0.35), 1);
  assert.equal(GR.smoothstep(0.04, 0.35, 1.0), 1);
  const mid = GR.smoothstep(0.04, 0.35, 0.195);
  assert.ok(mid > 0.4 && mid < 0.6, `midpoint ≈ 0.5, got ${mid}`);
});
check("background haze fades to transparent, plume core stays opaque", () => {
  // 10x10 gradient 0..1; normalization is p2/p98 ≈ the same range.
  const values = [];
  for (let r = 0; r < 10; r++) {
    const row = [];
    for (let c = 0; c < 10; c++) row.push((r * 10 + c) / 99);
    values.push(row);
  }
  const buf = GR.gridToPixelBuffer({
    variable: "DUEXTTAU",
    lats: Array.from({ length: 10 }, (_, i) => i),
    lons: Array.from({ length: 10 }, (_, i) => i),
    values,
  });
  const alpha = (y, x) => buf.data[(y * 10 + x) * 4 + 3];
  // Interior low-value pixel (south-west, t≈0.1) -> nearly transparent.
  assert.ok(alpha(8, 1) < 60, `haze alpha ${alpha(8, 1)} should be faint`);
  // Interior high-value pixel (north-east, t≈0.8, 2px from the border so
  // the edge feather does not touch it) -> near full opacity.
  assert.ok(alpha(2, 7) > 200, `core alpha ${alpha(2, 7)} should be strong`);
});
check("bbox border pixels are feathered on larger grids", () => {
  const values = Array.from({ length: 10 }, () => Array(10).fill(0.8));
  const buf = GR.gridToPixelBuffer({
    variable: "DUEXTTAU",
    lats: Array.from({ length: 10 }, (_, i) => i),
    lons: Array.from({ length: 10 }, (_, i) => i),
    values,
  });
  const alpha = (y, x) => buf.data[(y * 10 + x) * 4 + 3];
  // t = 0.5 -> valueAlpha = 1, so border vs center differs only by feather.
  assert.equal(alpha(0, 5), 0, "outermost border row is fully feathered out");
  assert.ok(alpha(1, 5) < alpha(5, 5), "feather ramp rises toward the interior");
  assert.ok(alpha(5, 5) > 200, "interior pixel keeps full alpha");
});
check("gridToPixelBuffer honors a fixed vmin/vmax override", () => {
  const mk = (v) => ({
    variable: "DUEXTTAU",
    lats: [0, 1],
    lons: [0, 1],
    values: [
      [v, v],
      [v, v],
    ],
  });
  // Same mid value, different fixed ranges -> different colors/legend.
  const lo = GR.gridToPixelBuffer(mk(5), { vmin: 0, vmax: 10 });
  const hi = GR.gridToPixelBuffer(mk(5), { vmin: 0, vmax: 100 });
  assert.equal(lo.vmin, 0);
  assert.equal(lo.vmax, 10);
  assert.equal(hi.vmax, 100);
  const px = (buf) => [buf.data[0], buf.data[1], buf.data[2]];
  assert.notDeepEqual(px(lo), px(hi), "t=0.5 vs t=0.05 must differ");
});
check("gridToPixelBuffer ignores a degenerate fixed range", () => {
  const grid = {
    variable: "DUEXTTAU",
    lats: [0, 1, 2],
    lons: [0, 1, 2],
    values: [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ],
  };
  const buf = GR.gridToPixelBuffer(grid, { vmin: 5, vmax: 5 });
  assert.ok(buf.vmax > buf.vmin, "falls back to data-driven range");
});

// partitionFrames: sparse archive days are skipped, order preserved.
check("partitionFrames splits kept and skipped days", () => {
  const g = (v) => ({ values: [[v]] });
  const r = GR.partitionFrames(
    [g(1), null, g(3)],
    ["2026-07-01", "2026-07-02", "2026-07-03"]
  );
  assert.deepEqual(
    r.kept.map((k) => k.date),
    ["2026-07-01", "2026-07-03"]
  );
  assert.equal(r.kept[0].grid.values[0][0], 1);
  assert.deepEqual(r.skipped, ["2026-07-02"]);
});
check("partitionFrames all missing", () => {
  const r = GR.partitionFrames([null, null], ["2026-07-01", "2026-07-02"]);
  assert.deepEqual(r.kept, []);
  assert.deepEqual(r.skipped, ["2026-07-01", "2026-07-02"]);
});
check("partitionFrames none missing", () => {
  const g = { values: [[1]] };
  const r = GR.partitionFrames([g], ["2026-07-01"]);
  assert.deepEqual(r.skipped, []);
  assert.equal(r.kept.length, 1);
});

console.log(`\n${n} tests passed.`);
