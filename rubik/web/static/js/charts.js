
export function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function setup(canvas) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function hexToRgb(h) {
  const v = h.replace("#", "");
  const n = parseInt(v.length === 3 ? v.split("").map((c) => c + c).join("") : v, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function ramp(stops) {
  const rgb = stops.map(hexToRgb);
  return (t) => {
    t = Math.max(0, Math.min(1, t));
    const x = t * (rgb.length - 1), i = Math.min(Math.floor(x), rgb.length - 2), f = x - i;
    const c = rgb[i].map((a, k) => Math.round(a + (rgb[i + 1][k] - a) * f));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  };
}

export const heatRamp = ramp(["#141826", "#3b2a6b", "#8f3f97", "#e0607e", "#ffb05c", "#fff1a8"]);
export const mapRamp = ramp(["#3cd3ff", "#8f7dff", "#ff5ec4", "#ffb347"]);
const divRamp = ramp(["#2f7bff", "#1a2a55", "#141823", "#5a2a14", "#ff8a1f"]);
const divRampLight = ramp(["#1f5fe0", "#a9c3f5", "#f3f4f7", "#f7c9a3", "#e66a00"]);

const tooltip = () => document.getElementById("tooltip");
export function showTip(x, y, html) {
  const t = tooltip();
  t.innerHTML = html;
  t.hidden = false;
  const r = t.getBoundingClientRect();
  let left = x + 14, top = y + 14;
  if (left + r.width > window.innerWidth - 8) left = x - r.width - 14;
  if (top + r.height > window.innerHeight - 8) top = y - r.height - 14;
  t.style.left = left + "px";
  t.style.top = top + "px";
}
export function hideTip() { tooltip().hidden = true; }

function isLight() {
  const t = document.documentElement.dataset.theme;
  if (t) return t === "light";
  return window.matchMedia("(prefers-color-scheme: light)").matches;
}

export function drawProbs(canvas, probs, chosen, meta, message) {
  const { ctx, w, h } = setup(canvas);
  if (message) {
    ctx.fillStyle = css("--good");
    ctx.font = `600 14px ${css("--font")}`;
    ctx.textAlign = "center";
    ctx.fillText("✓ " + message, w / 2, h / 2);
    return;
  }
  const names = meta.move_names, top = 16, bottom = 22, gap = 10;
  const groupW = (w - gap * 5) / 6, barW = (groupW - 4) / 2;
  const plotH = h - top - bottom;
  ctx.strokeStyle = css("--grid");
  ctx.lineWidth = 1;
  for (const f of [0.25, 0.5, 0.75, 1]) {
    const y = top + plotH * (1 - f) + 0.5;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }
  for (let a = 0; a < 12; a++) {
    const f = a >> 1, x = f * (groupW + gap) + (a & 1) * (barW + 4);
    const p = probs ? probs[a] : 0;
    const bh = Math.max(p * plotH, probs ? 1.5 : 0);
    const col = meta.face_hex[f];
    ctx.globalAlpha = a === chosen ? 1 : 0.55;
    ctx.fillStyle = col;
    roundRect(ctx, x, top + plotH - bh, barW, bh, 4);
    ctx.fill();
    ctx.globalAlpha = 1;
    if (a === chosen) {
      ctx.strokeStyle = css("--text");
      ctx.lineWidth = 1.5;
      roundRect(ctx, x - 1, top + plotH - bh - 1, barW + 2, bh + 2, 5);
      ctx.stroke();
    }
    ctx.fillStyle = a === chosen ? css("--text") : css("--text-2");
    ctx.font = `${a === chosen ? 700 : 500} 11px ${css("--mono")}`;
    ctx.textAlign = "center";
    ctx.fillText(names[a], x + barW / 2, h - 6);
    if (probs && p >= 0.06) {
      ctx.fillStyle = css("--text-2");
      ctx.font = `500 10px ${css("--mono")}`;
      ctx.fillText(Math.round(p * 100) + "", x + barW / 2, top + plotH - bh - 4);
    }
  }
}

function roundRect(ctx, x, y, w, h, r) {
  r = Math.max(0, Math.min(r, w / 2, Math.abs(h) / 2));
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

export function drawSpark(canvas, est, real, cur) {
  const { ctx, w, h } = setup(canvas);
  const n = Math.max(est.length, 2), pad = { l: 22, r: 6, t: 8, b: 14 };
  const vals = est.concat(real).filter((v) => v != null && isFinite(v));
  const ymax = Math.max(5, ...vals) * 1.08;
  const X = (i) => pad.l + (w - pad.l - pad.r) * (i / (n - 1));
  const Y = (v) => pad.t + (h - pad.t - pad.b) * (1 - v / ymax);
  ctx.strokeStyle = css("--grid");
  ctx.fillStyle = css("--tick");
  ctx.font = `10px ${css("--mono")}`;
  ctx.textAlign = "right";
  for (const v of niceTicks(0, ymax, 3)) {
    ctx.beginPath(); ctx.moveTo(pad.l, Y(v) + 0.5); ctx.lineTo(w - pad.r, Y(v) + 0.5); ctx.stroke();
    ctx.fillText(v, pad.l - 4, Y(v) + 3);
  }
  const line = (arr, color, dash, width) => {
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash);
    ctx.beginPath();
    let started = false;
    arr.forEach((v, i) => {
      if (v == null || !isFinite(v)) { started = false; return; }
      if (!started) { ctx.moveTo(X(i), Y(v)); started = true; } else ctx.lineTo(X(i), Y(v));
    });
    ctx.stroke(); ctx.setLineDash([]);
  };
  if (real && real.length) line(real, css("--text-2"), [4, 3], 1.4);
  line(est, css("--accent-2"), [], 2);
  if (cur != null && cur >= 0 && cur < est.length) {
    ctx.strokeStyle = css("--border-strong");
    ctx.beginPath(); ctx.moveTo(X(cur) + 0.5, pad.t); ctx.lineTo(X(cur) + 0.5, h - pad.b); ctx.stroke();
    if (est[cur] != null) {
      ctx.fillStyle = css("--accent-2");
      ctx.beginPath(); ctx.arc(X(cur), Y(est[cur]), 3.5, 0, Math.PI * 2); ctx.fill();
    }
  }
}

export function drawMemory(canvas, vec, lim = 1) {
  const { ctx, w, h } = setup(canvas);
  if (!vec) return;
  const n = vec.length, cols = n >= 512 ? 32 : Math.ceil(Math.sqrt(n * 2)), rows = Math.ceil(n / cols);
  const cw = w / cols, ch = h / rows, rr = isLight() ? divRampLight : divRamp;
  for (let i = 0; i < n; i++) {
    const r = Math.floor(i / cols), c = i % cols;
    ctx.fillStyle = rr(0.5 + 0.5 * Math.max(-1, Math.min(1, vec[i] / lim)));
    ctx.fillRect(c * cw + 0.5, r * ch + 0.5, cw - 1, ch - 1);
  }
}

export function drawMap(canvas, map, traj, cur) {
  const { ctx, w, h } = setup(canvas);
  if (!map) return;
  const pad = 12, [x0, y0] = map.lo, [x1, y1] = map.hi;
  const X = (x) => pad + (w - 2 * pad) * ((x - x0) / (x1 - x0 || 1));
  const Y = (y) => h - pad - (h - 2 * pad) * ((y - y0) / (y1 - y0 || 1));
  const maxd = 26;
  for (const [x, y, d] of map.points) {
    ctx.fillStyle = d < 0 ? css("--tick") : mapRamp(d / maxd);
    ctx.globalAlpha = d < 0 ? 0.25 : 0.55;
    ctx.fillRect(X(x) - 1, Y(y) - 1, 2.2, 2.2);
  }
  ctx.globalAlpha = 1;
  if (!traj || !traj.length) return;
  const upto = cur == null ? traj.length - 1 : Math.min(cur, traj.length - 1);
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.strokeStyle = css("--text");
  ctx.globalAlpha = 0.85;
  ctx.beginPath();
  for (let i = 0; i <= upto; i++) {
    const [x, y] = traj[i];
    if (i === 0) ctx.moveTo(X(x), Y(y)); else ctx.lineTo(X(x), Y(y));
  }
  ctx.stroke();
  ctx.globalAlpha = 1;
  for (let i = 0; i <= upto; i++) {
    const [x, y] = traj[i];
    ctx.fillStyle = i === upto ? css("--text") : css("--text-2");
    ctx.beginPath(); ctx.arc(X(x), Y(y), i === upto ? 5 : 2.2, 0, Math.PI * 2); ctx.fill();
  }
  const [hx, hy] = traj[upto];
  ctx.strokeStyle = css("--accent-2");
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(X(hx), Y(hy), 9, 0, Math.PI * 2); ctx.stroke();
  const [sx, sy] = traj[0];
  ctx.fillStyle = css("--text-2");
  ctx.font = `600 10px ${css("--font")}`;
  ctx.fillText("start", X(sx) + 7, Y(sy) - 6);
}

export function niceTicks(lo, hi, n = 4) {
  if (!(hi > lo)) return [lo];
  const span = hi - lo, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const err = step0 / mag;
  const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

function tickFmt(ticks) {
  const step = ticks.length > 1 ? Math.abs(ticks[1] - ticks[0]) : 1;
  const dec = Math.max(0, Math.min(4, -Math.floor(Math.log10(step) + 1e-9)));
  return (v) => (Math.abs(v) >= 10000 ? (v / 1000).toFixed(0) + "k" : v.toFixed(dec));
}

const fmt = (v) => (v == null || !isFinite(v) ? "—" : Math.abs(v) >= 1000 ? (v / 1000).toFixed(1) + "k" : Math.abs(v) >= 10 ? v.toFixed(1) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(3));

export function lineChart(canvas, opt) {
  canvas._opt = opt;
  if (!canvas._hover) {
    canvas._hover = true;
    canvas.addEventListener("mousemove", (e) => { canvas._mx = e.offsetX; renderLine(canvas, e); });
    canvas.addEventListener("mouseleave", () => { canvas._mx = null; hideTip(); renderLine(canvas); });
  }
  renderLine(canvas);
}

function renderLine(canvas, ev) {
  const opt = canvas._opt;
  if (!opt) return;
  const { ctx, w, h } = setup(canvas);
  const pad = { l: 40, r: opt.right ? 40 : 10, t: 10, b: 24 };
  const all = opt.series.flatMap((s) => s.pts);
  if (!all.length) {
    ctx.fillStyle = css("--muted");
    ctx.font = `12px ${css("--font")}`;
    ctx.textAlign = "center";
    ctx.fillText(opt.empty || "no data yet", w / 2, h / 2);
    return;
  }
  const xs = all.map((p) => p[0]);
  const x0 = opt.xMin ?? Math.min(...xs), x1 = opt.xMax ?? Math.max(...xs);
  const ranges = {};
  for (const ax of ["left", "right"]) {
    const ys = opt.series.filter((s) => (s.axis || "left") === ax).flatMap((s) => s.pts.map((p) => p[1])).filter((v) => v != null && isFinite(v));
    if (!ys.length) continue;
    let lo = opt[ax + "Min"] ?? Math.min(...ys), hi = opt[ax + "Max"] ?? Math.max(...ys);
    if (hi - lo < 1e-9) { hi = lo + 1; }
    if (opt[ax + "Min"] == null) lo -= (hi - lo) * 0.05;
    if (opt[ax + "Max"] == null) hi += (hi - lo) * 0.05;
    ranges[ax] = [lo, hi];
  }
  const X = (x) => pad.l + (w - pad.l - pad.r) * ((x - x0) / (x1 - x0 || 1));
  const Yf = (ax) => (y) => pad.t + (h - pad.t - pad.b) * (1 - (y - ranges[ax][0]) / (ranges[ax][1] - ranges[ax][0]));
  ctx.font = `10px ${css("--mono")}`;
  ctx.lineWidth = 1;
  for (const ax of Object.keys(ranges)) {
    const Y = Yf(ax);
    const ticks = niceTicks(ranges[ax][0], ranges[ax][1], 4), tf = tickFmt(ticks);
    for (const v of ticks) {
      if (ax === "left") {
        ctx.strokeStyle = css("--grid");
        ctx.beginPath(); ctx.moveTo(pad.l, Y(v) + 0.5); ctx.lineTo(w - pad.r, Y(v) + 0.5); ctx.stroke();
      }
      ctx.fillStyle = css("--tick");
      ctx.textAlign = ax === "left" ? "right" : "left";
      const lab = opt[ax + "Fmt"] ? opt[ax + "Fmt"](v) : tf(v);
      ctx.fillText(lab, ax === "left" ? pad.l - 5 : w - pad.r + 5, Y(v) + 3);
    }
  }
  ctx.textAlign = "center";
  ctx.fillStyle = css("--tick");
  const xt = niceTicks(x0, x1, 5), xtf = tickFmt(xt);
  for (const v of xt) ctx.fillText(opt.xTick ? opt.xTick(v) : opt.xFmt ? opt.xFmt(v) : xtf(v), X(v), h - 7);
  for (const m of opt.marks || []) {
    if (!(m.x > x0 && m.x < x1)) continue;
    const xm = Math.round(X(m.x)) + 0.5;
    ctx.strokeStyle = css("--text-2");
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(xm, pad.t); ctx.lineTo(xm, h - pad.b); ctx.stroke();
    ctx.setLineDash([]);
    if (m.label) {
      ctx.fillStyle = css("--text-2");
      ctx.textAlign = "left";
      ctx.fillText(m.label, xm + 4, pad.t - 2);
    }
  }
  ctx.textAlign = "center";
  for (const s of opt.series) {
    if (!s.pts.length) continue;
    const Y = Yf(s.axis || "left");
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.width || 1.8;
    ctx.setLineDash(s.dash || []);
    ctx.beginPath();
    let started = false;
    for (const [x, y] of s.pts) {
      if (y == null || !isFinite(y)) { started = false; continue; }
      if (!started) { ctx.moveTo(X(x), Y(y)); started = true; } else ctx.lineTo(X(x), Y(y));
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }
  if (canvas._mx != null && ev) {
    const xm = x0 + (x1 - x0) * ((canvas._mx - pad.l) / (w - pad.l - pad.r));
    let xv = all[0][0];
    for (const p of all) if (Math.abs(p[0] - xm) < Math.abs(xv - xm)) xv = p[0];
    const tol = 1e-9 * Math.max(1, Math.abs(x1 - x0));
    ctx.strokeStyle = css("--border-strong");
    ctx.beginPath(); ctx.moveTo(Math.round(X(xv)) + 0.5, pad.t); ctx.lineTo(Math.round(X(xv)) + 0.5, h - pad.b); ctx.stroke();
    const rows = [];
    for (const s of opt.series) {
      const best = s.pts.find((p) => Math.abs(p[0] - xv) <= tol);
      if (!best) continue;
      const Y = Yf(s.axis || "left");
      if (best[1] != null && isFinite(best[1])) {
        ctx.fillStyle = s.color;
        ctx.beginPath(); ctx.arc(X(best[0]), Y(best[1]), 3, 0, Math.PI * 2); ctx.fill();
      }
      rows.push(`<span style="color:${s.color}">●</span> ${s.name}: <b>${s.fmt ? s.fmt(best[1]) : fmt(best[1])}</b>`);
    }
    const head = opt.xFmt ? opt.xFmt(xv) : fmt(xv);
    showTip(ev.clientX, ev.clientY, `<div style="color:var(--muted)">${opt.xName || "x"} ${head}</div>${rows.join("<br>")}`);
  }
}

export function heatmap(canvas, opt) {
  canvas._hopt = opt;
  if (!canvas._hhover) {
    canvas._hhover = true;
    canvas.addEventListener("mousemove", (e) => renderHeat(canvas, e));
    canvas.addEventListener("mouseleave", () => { hideTip(); renderHeat(canvas); });
  }
  renderHeat(canvas);
}

function renderHeat(canvas, ev) {
  const opt = canvas._hopt;
  const { ctx, w, h } = setup(canvas);
  const R = opt.rowLabels.length, Cn = opt.values.length ? opt.values[0].length : 0;
  const pad = { l: 52, r: 8, t: 6, b: 24 };
  if (!Cn) {
    ctx.fillStyle = css("--muted");
    ctx.font = `12px ${css("--font")}`;
    ctx.textAlign = "center";
    ctx.fillText(opt.empty || "no evaluations yet", w / 2, h / 2);
    return;
  }
  const cw = (w - pad.l - pad.r) / Cn, ch = (h - pad.t - pad.b) / R;
  for (let r = 0; r < R; r++) for (let c = 0; c < Cn; c++) {
    const v = opt.values[r][c];
    ctx.fillStyle = v == null ? css("--panel-2") : heatRamp(v);
    ctx.fillRect(pad.l + c * cw, pad.t + (R - 1 - r) * ch, Math.ceil(cw) + 0.5, Math.ceil(ch) + 0.5);
  }
  ctx.font = `10px ${css("--mono")}`;
  for (const m of opt.colMarks || []) {
    const xm = Math.round(pad.l + m.col * cw) + 0.5;
    ctx.strokeStyle = "rgba(255,255,255,.85)";
    ctx.lineWidth = 1.2;
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(xm, pad.t); ctx.lineTo(xm, h - pad.b); ctx.stroke();
    ctx.setLineDash([]);
    if (m.label) {
      ctx.fillStyle = "rgba(255,255,255,.9)";
      ctx.textAlign = "left";
      ctx.fillText(m.label, xm + 5, pad.t + 11);
    }
  }
  ctx.lineWidth = 1;
  ctx.fillStyle = css("--tick");
  ctx.textAlign = "right";
  const every = Math.max(1, Math.ceil(R / 12));
  for (let r = 0; r < R; r++) {
    if (r % every && r !== R - 1) continue;
    ctx.fillText(opt.rowLabels[r], pad.l - 6, pad.t + (R - 1 - r) * ch + ch / 2 + 3);
  }
  ctx.textAlign = "center";
  const ce = Math.max(1, Math.ceil(Cn / 8));
  for (let c = 0; c < Cn; c += ce) ctx.fillText(opt.colLabels[c], pad.l + c * cw + cw / 2, h - 7);
  if (opt.rowTitle) {
    ctx.save(); ctx.translate(11, pad.t + (h - pad.t - pad.b) / 2); ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center"; ctx.fillText(opt.rowTitle, 0, 0); ctx.restore();
  }
  if (ev) {
    const c = Math.floor((ev.offsetX - pad.l) / cw), r = R - 1 - Math.floor((ev.offsetY - pad.t) / ch);
    if (c >= 0 && c < Cn && r >= 0 && r < R) {
      ctx.strokeStyle = css("--text");
      ctx.lineWidth = 1.5;
      ctx.strokeRect(pad.l + c * cw, pad.t + (R - 1 - r) * ch, cw, ch);
      const v = opt.values[r][c];
      showTip(ev.clientX, ev.clientY, `${opt.rowName || "row"} <b>${opt.rowLabels[r]}</b> · ${opt.colName || "col"} <b>${opt.colLabels[c]}</b><br>${opt.valueName || "value"}: <b>${v == null ? "—" : Math.round(v * 100) + " %"}</b>`);
    } else hideTip();
  }
}
