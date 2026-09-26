/* Binary hourly contract helpers, usable in browser and Node tests. */
(function (root) {
  function decode(buffer, manifest, count) {
    const size = manifest.nx * manifest.ny;
    if (buffer.byteLength !== size * count * 4) throw new Error('Incomplete hourly batch.');
    const view = new DataView(buffer);
    const values = new Float32Array(size * count);
    for (let i = 0; i < values.length; i++) values[i] = view.getFloat32(i * 4, true);
    return Array.from({ length: count }, (_, i) => values.subarray(i * size, (i + 1) * size));
  }
  function grid(manifest, values, timestamp) {
    return { ...manifest, values: Array.from({ length: manifest.ny }, (_, r) =>
      Array.from(values.subarray(r * manifest.nx, (r + 1) * manifest.nx))),
      time_start: timestamp, time_end: timestamp };
  }
  const api = { decode, grid };
  if (typeof module !== 'undefined') module.exports = api;
  else root.HourlyFrames = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
