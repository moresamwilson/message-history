// ===========================================================================
// Shared colour tokens for every chart. Tweak a value here and every chart
// that references it updates. Included as a plain <script> (not fetched) so
// the charts work straight from file:// with no server.
//
// Base palette is Paul Tol's colourblind-safe set. Mum & Dad appear in two
// forms: flat categorical colours (side-by-side swatches keep a slight
// lightness difference so they still separate in greyscale), and sequential
// ramps for the heatmaps, whose dark ends are pinned to the same luminance
// so a full Mum cell and a full Dad cell read as equally intense.
// ===========================================================================
const CHART_COLORS = {
  // --- People ---------------------------------------------------------------
  mum:  "#b95264",   // muted red
  dad:  "#003f7e",   // deep blue
  self: "#6b7280",   // my own messages — a neutral grey baseline in every panel

  // Sequential light→dark ramp parameters (CIE-LUV), consumed by chartRamp()
  mumScale: { hue:   2.6, chromaMax: 63.5, lLow: 97, lHigh: 27.7, power: 1.15 },
  dadScale: { hue: 254.1, chromaMax: 54.3, lLow: 97, lHigh: 26.9, power: 1.15 },

  // --- Structural / non-data ink -------------------------------------------
  ink:        "#1d2430",   // titles, zero baselines, dark text
  muted:      "#6b7280",   // subtitles, axis labels, secondary text
  grid:       "#cfd3da",   // faint reference gridlines
  frame:      "#000000",   // outer canvas outline
  background: "#ffffff",   // page + stage background
};

// ===========================================================================
// Ramp machinery: chartRamp(scale) turns a *Scale object into
//   t ∈ [0,1] -> "rgb(r,g,b)"  (light at t=0, dark at t=1)
// built in CIE-LUV so equal t means equal perceived darkness — the property
// the heatmaps rely on. Replicates R's grDevices::hcl() sequential ramps.
// ===========================================================================
const _D65 = { Xn: 95.047, Yn: 100.0, Zn: 108.883 };
const _UN = 4 * _D65.Xn / (_D65.Xn + 15 * _D65.Yn + 3 * _D65.Zn);
const _VN = 9 * _D65.Yn / (_D65.Xn + 15 * _D65.Yn + 3 * _D65.Zn);

function _gamma(c) {
  c = c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
  return Math.max(0, Math.min(1, c));
}

// polar LUV (h degrees, chroma, luminance) -> CSS rgb()
function hcl2rgb(h, chroma, L) {
  const hr = h * Math.PI / 180;
  const U = chroma * Math.cos(hr), V = chroma * Math.sin(hr);
  let X, Y, Z;
  if (L <= 0) { X = Y = Z = 0; }
  else {
    Y = L > 8 ? _D65.Yn * Math.pow((L + 16) / 116, 3) : _D65.Yn * L / 903.3;
    const up = U / (13 * L) + _UN, vp = V / (13 * L) + _VN;
    X = Y * 9 * up / (4 * vp);
    Z = Y * (12 - 3 * up - 20 * vp) / (4 * vp);
  }
  X /= 100; Y /= 100; Z /= 100;
  const r = _gamma( 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z);
  const g = _gamma(-0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z);
  const b = _gamma( 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z);
  return `rgb(${Math.round(r*255)},${Math.round(g*255)},${Math.round(b*255)})`;
}

// Build a t -> colour ramp from a *Scale param object.
function chartRamp({ hue, chromaMax = 62, lLow = 97, lHigh = 33, power = 1.15 }) {
  return t => {
    const tt = Math.pow(Math.max(0, Math.min(1, t)), power);
    return hcl2rgb(hue, chromaMax * tt, lLow + (lHigh - lLow) * tt);
  };
}

// Ready-made ramps keyed the way the charts label people.
const CHART_RAMPS = {
  Mum: chartRamp(CHART_COLORS.mumScale),
  Dad: chartRamp(CHART_COLORS.dadScale),
};

if (typeof module !== "undefined" && module.exports) {
  module.exports = { CHART_COLORS, chartRamp, CHART_RAMPS, hcl2rgb };
}
