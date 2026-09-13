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
  assert.deepEqual(GR.sampleColormap(GR.COLORMAPS.dust, 0), [255, 255, 204]);
  assert.deepEqual(GR.sampleColormap(GR.COLORMAPS.dust, 1), [128, 0, 38]);
});
check("viridis colormap endpoints", () => {
  assert.deepEqual(GR.sampleColormap(GR.COLORMAPS.viridis, 0), [68, 1, 84]);
  assert.deepEqual(GR.sampleColormap(GR.COLORMAPS.viridis, 1), [253, 231, 37]);
});
check("colormap midpoint interpolates between stops", () => {
  const [r, g, b] = GR.sampleColormap(GR.COLORMAPS.dust, 0.5);
  assert.deepEqual([r, g, b], [253, 141, 60]); // #fd8d3c exactly at t=0.5
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
  assert.deepEqual(top, [128, 0, 38], "top row = dust(1.0)");
  assert.deepEqual(bottom, [255, 255, 204], "bottom row = dust(0.0)");
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
  assert.ok(g.includes("#ffffcc") && g.includes("#800026"), g);
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

console.log(`\n${n} tests passed.`);
