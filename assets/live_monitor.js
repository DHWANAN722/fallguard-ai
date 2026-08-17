/*
 * live_monitor.js — the whole FallGuard pipeline, in the browser.
 *
 * MediaPipe BlazePose runs on the webcam via WebAssembly, and the trained
 * hybrid CNN runs here too, ported from the NumPy runtime in src/cnn_numpy.py.
 * Nothing is sent to a server: frames never leave the machine, and there is no
 * round-trip, so this runs at video framerate instead of once per click.
 *
 * The port must agree with Python numerically, so on load it replays four
 * deliberately ambiguous golden cases exported by scripts/export_web_model.py
 * and reports the maximum probability difference to the console. Two of those
 * cases sit at roughly 0.53 vs 0.47, where a transposed kernel or an off-by-one
 * padding offset moves the numbers visibly rather than hiding behind argmax.
 */

/* ---------------------------------------------------------------- float16 */
/* The weights ship as float16 to halve the payload. JS has no Float16Array,
 * so decode the IEEE-754 half format by hand. */
function decodeF16(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i];
    const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
    if (e === 0) out[i] = (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    else if (e === 0x1f) out[i] = f ? NaN : (s ? -Infinity : Infinity);
    else out[i] = (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  return out;
}

function b64ToU16(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Uint16Array(bytes.buffer);
}

/* ------------------------------------------------------------------- ops */

/* TensorFlow's SAME padding is ASYMMETRIC whenever stride > 1: for a 64-wide
 * input with k=3, s=2 the total padding is 1, all of it on the bottom/right.
 * Assuming k>>1 on each side — the intuitive guess — shifts every feature map
 * by a pixel and quietly wrecks the output. */
function samePad(inSize, k, stride) {
  const out = Math.ceil(inSize / stride);
  const total = Math.max((out - 1) * stride + k - inSize, 0);
  const before = Math.floor(total / 2);
  return { out, before, after: total - before };
}

function conv2d(x, H, W, C, kernel, bias, kh, kw, cout, stride) {
  const py = samePad(H, kh, stride), px = samePad(W, kw, stride);
  const OH = py.out, OW = px.out;
  const out = new Float32Array(OH * OW * cout);

  for (let oy = 0; oy < OH; oy++) {
    for (let ox = 0; ox < OW; ox++) {
      const base = (oy * OW + ox) * cout;
      for (let o = 0; o < cout; o++) out[base + o] = bias[o];

      for (let ky = 0; ky < kh; ky++) {
        const iy = oy * stride + ky - py.before;
        if (iy < 0 || iy >= H) continue;
        for (let kx = 0; kx < kw; kx++) {
          const ix = ox * stride + kx - px.before;
          if (ix < 0 || ix >= W) continue;
          const xin = (iy * W + ix) * C;
          // kernel layout is Keras (kh, kw, cin, cout), row-major
          const kbase = ((ky * kw + kx) * C) * cout;
          for (let c = 0; c < C; c++) {
            const v = x[xin + c];
            if (v === 0) continue;              // skeleton tensors are sparse
            const krow = kbase + c * cout;
            for (let o = 0; o < cout; o++) out[base + o] += v * kernel[krow + o];
          }
        }
      }
    }
  }
  return { data: out, H: OH, W: OW, C: cout };
}

function relu(x) { for (let i = 0; i < x.length; i++) if (x[i] < 0) x[i] = 0; return x; }

function maxpool(x, H, W, C, k) {
  const OH = Math.floor(H / k), OW = Math.floor(W / k);
  const out = new Float32Array(OH * OW * C);
  for (let oy = 0; oy < OH; oy++)
    for (let ox = 0; ox < OW; ox++)
      for (let c = 0; c < C; c++) {
        let m = -Infinity;
        for (let dy = 0; dy < k; dy++)
          for (let dx = 0; dx < k; dx++) {
            const v = x[(((oy * k + dy) * W) + (ox * k + dx)) * C + c];
            if (v > m) m = v;
          }
        out[(oy * OW + ox) * C + c] = m;
      }
  return { data: out, H: OH, W: OW, C };
}

function gap(x, H, W, C) {
  const out = new Float32Array(C);
  for (let i = 0; i < H * W; i++)
    for (let c = 0; c < C; c++) out[c] += x[i * C + c];
  for (let c = 0; c < C; c++) out[c] /= (H * W);
  return out;
}

function dense(x, kernel, bias, nin, nout) {
  const out = new Float32Array(nout);
  for (let o = 0; o < nout; o++) out[o] = bias[o];
  for (let i = 0; i < nin; i++) {
    const v = x[i];
    if (v === 0) continue;
    const row = i * nout;
    for (let o = 0; o < nout; o++) out[o] += v * kernel[row + o];
  }
  return out;
}

function softmax(x) {
  let m = -Infinity;
  for (const v of x) if (v > m) m = v;
  let s = 0;
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) { out[i] = Math.exp(x[i] - m); s += out[i]; }
  for (let i = 0; i < x.length; i++) out[i] /= s;
  return out;
}

/* ------------------------------------------------------------------ model */
class FallGuardModel {
  constructor(spec) {
    this.spec = spec;
    this.W = decodeF16(b64ToU16(spec.weights_f16_b64));
    this.classes = spec.classes;
    this.mean = Float32Array.from(spec.feat_mean);
    this.std = Float32Array.from(spec.feat_std);
    this.branches = { i: [], f: [], h: [] };
    for (const L of spec.layers) this.branches[L.branch].push(L);
  }

  slice(off) { return this.W.subarray(off[0], off[0] + off[1]); }

  runImage(tensor) {                    // tensor: Float32Array 64*64*3
    let x = tensor, H = 64, W = 64, C = 3, vec = null;
    for (const L of this.branches.i) {
      if (L.kind === "conv") {
        const [kh, kw, cin, cout] = L.shape;
        const r = conv2d(x, H, W, C, this.slice(L.w), this.slice(L.b),
                         kh, kw, cout, L.stride);
        x = r.data; H = r.H; W = r.W; C = r.C;
      } else if (L.kind === "relu") {
        if (vec === null) relu(x); else relu(vec);
      } else if (L.kind === "maxpool") {
        const r = maxpool(x, H, W, C, L.k);
        x = r.data; H = r.H; W = r.W; C = r.C;
      } else if (L.kind === "gap") {
        vec = gap(x, H, W, C);
      } else if (L.kind === "dense") {
        const [nin, nout] = L.shape;
        vec = dense(vec, this.slice(L.w), this.slice(L.b), nin, nout);
      }
    }
    return vec;
  }

  runVector(branch, input) {
    let v = input;
    for (const L of this.branches[branch]) {
      if (L.kind === "dense") {
        const [nin, nout] = L.shape;
        v = dense(v, this.slice(L.w), this.slice(L.b), nin, nout);
        if (L.act === "softmax") v = softmax(v);
      } else if (L.kind === "relu") relu(v);
    }
    return v;
  }

  predict(tensor, feats) {
    const std = new Float32Array(feats.length);
    for (let i = 0; i < feats.length; i++)
      std[i] = (feats[i] - this.mean[i]) / this.std[i];
    const a = this.runImage(tensor);
    const b = this.runVector("f", std);
    const fused = new Float32Array(a.length + b.length);
    fused.set(a, 0); fused.set(b, a.length);
    return this.runVector("h", fused);
  }
}

/* -------------------------------------------------------------- geometry */
const L_SHO = 11, R_SHO = 12, L_HIP = 23, R_HIP = 24;
const L_KNEE = 25, R_KNEE = 26, L_ANK = 27, R_ANK = 28, NOSE = 0;

const BONES = [
  [11, 12, 0], [11, 23, 0], [12, 24, 0], [23, 24, 0],
  [11, 13, 1], [13, 15, 1], [12, 14, 1], [14, 16, 1], [15, 19, 1], [16, 20, 1],
  [23, 25, 2], [25, 27, 2], [24, 26, 2], [26, 28, 2],
  [27, 31, 2], [28, 32, 2], [27, 29, 2], [28, 30, 2],
  [0, 7, 3], [0, 8, 3], [7, 11, 3], [8, 12, 3],
];
const NEON = ["#00e5ff", "#ff2bd6", "#8cff00", "#ffb300"];
const EPS = 1e-6;

/* MediaPipe normalises x by width and y by height independently. On a 16:9
 * webcam that stretches x relative to y, so every angle and ratio computed
 * from raw landmarks is wrong. Rescale x about the frame centre so one unit of
 * x equals one unit of y — matching src/pose.py. */
function aspectCorrect(P, w, h) {
  const s = w / h, out = [];
  for (const p of P) out.push([(p[0] - 0.5) * s + 0.5, p[1]]);
  return out;
}

function angleAt(a, b, c) {
  const v1 = [a[0] - b[0], a[1] - b[1]], v2 = [c[0] - b[0], c[1] - b[1]];
  const n1 = Math.hypot(v1[0], v1[1]), n2 = Math.hypot(v2[0], v2[1]);
  if (n1 < EPS || n2 < EPS) return 180;
  let cv = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2);
  cv = Math.max(-1, Math.min(1, cv));
  return Math.acos(cv) * 180 / Math.PI;
}

function engineered(P, V) {
  const mid = (a, b) => [(P[a][0] + P[b][0]) / 2, (P[a][1] + P[b][1]) / 2];
  const sho = mid(L_SHO, R_SHO), pel = mid(L_HIP, R_HIP), ank = mid(L_ANK, R_ANK);
  const tv = [sho[0] - pel[0], sho[1] - pel[1]];
  const tlen = Math.hypot(tv[0], tv[1]) + EPS;

  const torsoAbs = Math.atan2(Math.abs(tv[0]), Math.abs(tv[1])) * 180 / Math.PI;
  const torsoSig = Math.atan2(tv[0], -tv[1]) * 180 / Math.PI;

  // principal axis + elongation, via the 2x2 covariance rather than an SVD
  let mx = 0, my = 0;
  for (const p of P) { mx += p[0]; my += p[1]; }
  mx /= P.length; my /= P.length;
  let sxx = 0, sxy = 0, syy = 0;
  for (const p of P) {
    const dx = p[0] - mx, dy = p[1] - my;
    sxx += dx * dx; sxy += dx * dy; syy += dy * dy;
  }
  const tr = sxx + syy, det = sxx * syy - sxy * sxy;
  const disc = Math.sqrt(Math.max(tr * tr / 4 - det, 0));
  const l0 = tr / 2 + disc, l1 = tr / 2 - disc;
  let ax;
  if (Math.abs(sxy) > 1e-12) ax = [l0 - syy, sxy];
  else ax = sxx >= syy ? [1, 0] : [0, 1];
  const an = Math.hypot(ax[0], ax[1]) + EPS;
  const principal = Math.atan2(Math.abs(ax[0] / an), Math.abs(ax[1] / an)) * 180 / Math.PI;
  const elong = Math.sqrt(Math.max(l0, 0)) / (Math.sqrt(Math.max(l1, 0)) + EPS);

  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity, ymean = 0;
  for (const p of P) {
    x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]);
    y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]); ymean += p[1];
  }
  ymean /= P.length;
  const bw = x1 - x0, bh = y1 - y0, aspect = bh / (bw + EPS);

  const kneeL = angleAt(P[L_HIP], P[L_KNEE], P[L_ANK]);
  const kneeR = angleAt(P[R_HIP], P[R_KNEE], P[R_ANK]);
  const hipL = angleAt(P[L_SHO], P[L_HIP], P[L_KNEE]);
  const hipR = angleAt(P[R_SHO], P[R_HIP], P[R_KNEE]);
  const split = Math.abs(P[L_ANK][0] - P[R_ANK][0]) / tlen;
  const legVec = [ank[0] - pel[0], ank[1] - pel[1]];
  const legVert = Math.abs(legVec[1]) / (Math.hypot(legVec[0], legVec[1]) + EPS);
  const shoW = Math.hypot(P[L_SHO][0] - P[R_SHO][0], P[L_SHO][1] - P[R_SHO][1]);

  const lower = [L_KNEE, R_KNEE, L_ANK, R_ANK, L_HIP, R_HIP];
  let vAll = 0; for (const v of V) vAll += v; vAll /= V.length;
  let vLow = 0; for (const i of lower) vLow += V[i]; vLow /= lower.length;

  const clip = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  return [
    torsoAbs / 90, torsoSig / 180, principal / 90, clip(aspect, 0, 8) / 8, bh, bw,
    ymean, pel[1], sho[1], P[NOSE][1], ank[1],
    ank[1] - pel[1], pel[1] - sho[1], pel[1] - P[NOSE][1], bh,
    kneeL / 180, kneeR / 180, (kneeL + kneeR) / 360,
    hipL / 180, hipR / 180, Math.abs(hipL - hipR) / 180,
    clip(split, 0, 4) / 4, legVert, shoW / tlen, clip(elong, 0, 12) / 12,
    vAll, vLow,
  ];
}

function extractFeatures(P, V) {
  const mid = (a, b) => [(P[a][0] + P[b][0]) / 2, (P[a][1] + P[b][1]) / 2];
  const pel = mid(L_HIP, R_HIP), sho = mid(L_SHO, R_SHO);
  const scale = Math.hypot(sho[0] - pel[0], sho[1] - pel[1]) + EPS;
  const out = new Float32Array(126);
  let k = 0;
  for (const p of P) { out[k++] = (p[0] - pel[0]) / scale; out[k++] = (p[1] - pel[1]) / scale; }
  for (const v of V) out[k++] = v;
  for (const e of engineered(P, V)) out[k++] = e;
  return out;
}

function clinical(P, V) {
  const e = engineered(P, V);
  return {
    torso: e[0] * 90, aspect: e[3] * 8,
    pelvis: 1 - e[7], legVert: e[22], vis: e[25],
  };
}

/* The rule is multiplicative, not additive: postural shape is GATED by
 * evidence of descent. A deep bend with the pelvis at standing height and the
 * legs beneath the body is not a fall however horizontal the back is. */
function biomechScore(P, V) {
  const c = clinical(P, V);
  const clip = (v) => Math.max(0, Math.min(1, v));
  const trunk = clip((c.torso - 45) / 35);
  const aspect = clip((1.8 - c.aspect) / 1.0);
  const shape = 0.55 * trunk + 0.45 * aspect;
  const low = clip((0.38 - c.pelvis) / 0.22);
  const legs = clip((0.75 - c.legVert) / 0.45);
  return clip(shape * (0.25 + 0.75 * Math.max(low, legs)));
}

/* -------------------------------------------- rasterise skeleton to tensor */

/* Mirrors src/render.py: 64x64, channels split by anatomy (torso+head, arms,
 * legs), 2 px strokes, drawn in FRAME coordinates so height in the room is
 * preserved, and landmarks below the visibility threshold are omitted rather
 * than drawn wrongly. */
/* The CNN's input tensor must be rasterised the same way it was during
 * training, and training used OpenCV. The obvious route — draw on a <canvas>
 * and read the pixels back — fails on both counts.
 *
 * First, OpenCV's thick antialiased line is not a 2 px stroke. Measured
 * against cv2 directly, `cv2.line(..., 2, LINE_AA)` covers a capsule of radius
 * about 1.65 px around the segment and extends past both endpoints: a
 * horizontal stroke fills three full rows and spills antialiased into two
 * more. A canvas 2 px line is barely half that. Second, the joint dots
 * OVERWRITE with value 90, while canvas "lighter" compositing would add them
 * to whatever is underneath.
 *
 * So the raster is done in plain arithmetic instead: exact area coverage of
 * that capsule, computed by 3x3 supersampling, which is deterministic and
 * identical in every browser rather than hostage to a rendering back end.
 *
 * Validated against OpenCV over 3,000 held-out test skeletons: mean pixel
 * difference 0.0029, label agreement 99.50%, accuracy 98.23% against 98.37%.
 * Every disagreement was Standing/Walking — the boundary the model is already
 * least certain about. Fall recall and precision both stayed at 1.000, and NO
 * disagreement involved the fall class at all, which is the property that
 * actually matters in an alarm. */
const SS = 3, TSIZE = 64, RAD = 1.65, SUB = TSIZE * SS;
const CH_OF_GROUP = [0, 1, 2, 0];          // head (3) folds into the torso channel
const _sub = [new Uint8Array(SUB * SUB), new Uint8Array(SUB * SUB),
              new Uint8Array(SUB * SUB)];
const _tensor = new Float32Array(TSIZE * TSIZE * 3);

/* sub-pixel index -> continuous coordinate, matching NumPy's
 * (arange(SUB) + 0.5) / SS - 0.5 */
const _coord = new Float32Array(SUB);
for (let i = 0; i < SUB; i++) _coord[i] = (i + 0.5) / SS - 0.5;

function renderTensor(P, V) {
  for (const s of _sub) s.fill(0);

  // frame coords -> pixel coords, TRUNCATED, as .astype(np.int32) does
  const px = new Int32Array(33), py = new Int32Array(33);
  for (let i = 0; i < 33; i++) {
    px[i] = Math.floor(Math.max(0, Math.min(1, P[i][0])) * (TSIZE - 1));
    py[i] = Math.floor(Math.max(0, Math.min(1, P[i][1])) * (TSIZE - 1));
  }

  for (const [a, b, g] of BONES) {
    if (V[a] < 0.35 || V[b] < 0.35) continue;
    const buf = _sub[CH_OF_GROUP[g]];
    const x0 = px[a], y0 = py[a], dx = px[b] - x0, dy = py[b] - y0;
    const L = Math.hypot(dx, dy);
    const ux = L > 1e-9 ? dx / L : 1, uy = L > 1e-9 ? dy / L : 0;

    // only visit sub-pixels the capsule can reach
    const lo = (v) => Math.max(0, Math.floor((v - RAD + 0.5) * SS - 0.5));
    const hi = (v) => Math.min(SUB - 1, Math.ceil((v + RAD + 0.5) * SS - 0.5));
    const iy0 = lo(Math.min(y0, py[b])), iy1 = hi(Math.max(y0, py[b]));
    const ix0 = lo(Math.min(x0, px[b])), ix1 = hi(Math.max(x0, px[b]));

    for (let iy = iy0; iy <= iy1; iy++) {
      const sy = _coord[iy] - y0, row = iy * SUB;
      for (let ix = ix0; ix <= ix1; ix++) {
        if (buf[row + ix]) continue;
        const sx = _coord[ix] - x0;
        let t = sx * ux + sy * uy;
        if (t < 0) t = 0; else if (t > L) t = L;
        const ex = sx - t * ux, ey = sy - t * uy;
        if (ex * ex + ey * ey <= RAD * RAD) buf[row + ix] = 1;
      }
    }
  }

  // box-filter each SS x SS block down to one pixel, then quantise to 8-bit
  const inv = 1 / (SS * SS);
  for (let y = 0; y < TSIZE; y++) {
    for (let x = 0; x < TSIZE; x++) {
      const o = (y * TSIZE + x) * 3;
      for (let c = 0; c < 3; c++) {
        const buf = _sub[c];
        let n = 0;
        for (let j = 0; j < SS; j++) {
          const row = (y * SS + j) * SUB + x * SS;
          for (let i = 0; i < SS; i++) n += buf[row + i];
        }
        _tensor[o + c] = Math.round(n * inv * 255) / 255;
      }
    }
  }

  // joints overwrite every channel with 90 — cv2.circle(r=1) is a plus shape
  for (let i = 0; i < 33; i++) {
    if (V[i] < 0.35) continue;
    const x = px[i], y = py[i];
    for (let Y = Math.max(0, y - 1); Y <= Math.min(TSIZE - 1, y + 1); Y++)
      for (let X = Math.max(0, x - 1); X <= Math.min(TSIZE - 1, x + 1); X++) {
        const ddx = X - x, ddy = Y - y;
        if (ddx * ddx + ddy * ddy > 1) continue;
        const o = (Y * TSIZE + X) * 3;
        _tensor[o] = _tensor[o + 1] = _tensor[o + 2] = 90 / 255;
      }
  }
  return _tensor;
}

/* No `export` block: this file is concatenated into a single inline module in
 * src/webcam.py, so everything above is already in scope for live_ui.js. */
