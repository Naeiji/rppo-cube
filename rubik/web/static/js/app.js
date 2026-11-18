import { Cube3D } from "./cube3d.js";
import * as Ch from "./charts.js";

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const SPEEDS = [[750, 600], [480, 340], [300, 170], [170, 70], [90, 20]];
const SOLVED = Array.from({ length: 54 }, (_, i) => i);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const LN11 = Math.log(11);
const TOGO_CAP = 100;

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  let d = null;
  try { d = await r.json(); } catch (e) {}
  if (!r.ok) throw new Error((d && d.error) || `${r.status} ${r.statusText}`);
  return d;
}

function toast(msg, err = false) {
  const t = document.createElement("div");
  t.className = "toast" + (err ? " err" : "");
  t.textContent = msg;
  $("#toasts").appendChild(t);
  setTimeout(() => t.remove(), err ? 6000 : 3500);
}

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const fmtMovesM = (v) => (v >= 1000 ? (v / 1000).toFixed(2).replace(/\.?0+$/, "") + " B" : (v >= 10 || v === 0 ? Math.round(v) : v.toFixed(1)) + " M");
const fmtMoves = (samples) => fmtMovesM(samples / 1e6) + " moves";

const S = {
  meta: null,
  snapshot: null,
  cube: null,
  start: SOLVED.slice(),
  setup: { label: "solved cube", moves: [] },
  trace: null,
  pos: 0,
  playing: false,
  token: 0,
  playId: 0,
  stepping: false,
  gen: 0,
  busy: false,
  hint: null,
  map: null,
  memMode: "h",
  training: null,
  pollTimer: null,
};

const applyMoves = (st, moves) => moves.reduce((s, a) => S.meta.move_src[a].map((i) => s[i]), st);
const isSolved = (st) => st.every((v, i) => v === i);
const currentState = () => applyMoves(S.start, S.trace ? S.trace.moves.slice(0, S.pos) : []);
const speed = () => SPEEDS[+$("#speed").value - 1];

async function boot() {
  S.meta = await api("/api/meta");
  S.cube = new Cube3D($("#cube-canvas"), S.meta);
  buildMovePad();
  bindControls();
  window.addEventListener("hashchange", route);
  route();
  setStatus();
  renderTimeline();
  renderMind();
  window.APP_DEBUG = { S, PAINT, paintAt, applyMoves };
}

const withSnapshot = (body) => (S.snapshot == null ? body : { ...body, snapshot: S.snapshot });

function renderAgentChip() {
  const chip = $("#agent-chip");
  chip.hidden = S.snapshot == null;
  if (S.snapshot == null) return;
  const sn = S.training ? S.training.snapshots.find((x) => x.it === S.snapshot) : null;
  chip.textContent = (S.snapshot === 0 || !sn ? "untrained agent" : `agent after ${fmtMoves(sn.samples)}`) + "  ✕";
  chip.title = "Back to the trained agent";
}

async function setSnapshot(it) {
  await interrupt();
  clearTrace();
  S.snapshot = it;
  S.map = null;
  mapKey = null;
  renderAgentChip();
  renderMind();
  refreshHint();
}

let mapKey = null;
function ensureMap(version) {
  if (S.map && (!version || S.map.version === version)) return;
  const key = S.snapshot + "@" + (version || "");
  if (mapKey === key) return;
  mapKey = key;
  const snap = S.snapshot;
  api("/api/thoughtmap" + (snap == null ? "" : `?snapshot=${snap}`))
    .then((m) => { if (S.snapshot === snap) { S.map = m; renderMind(); } })
    .catch((e) => { if (S.snapshot === snap) toast("thought map: " + e.message, true); });
}

function buildMovePad() {
  const pad = $("#movepad");
  const order = [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11];
  for (const a of order) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = S.meta.move_names[a];
    b.style.setProperty("--fc", S.meta.face_hex[a >> 1]);
    b.title = `${S.meta.move_names[a]}: turn the ${S.meta.color_names[a >> 1]} face ${a & 1 ? "counter-clockwise" : "clockwise"}`;
    b.addEventListener("click", () => { stopDemo(); manualMove(a); });
    pad.appendChild(b);
  }
}

function bindControls() {
  const depth = $("#depth");
  const upd = () => ($("#depth-out").textContent = depth.value);
  depth.addEventListener("input", upd);
  upd();
  $("#btn-scramble").addEventListener("click", () => { stopDemo(); scramble({ depth: +depth.value }); });
  $("#btn-random").addEventListener("click", () => { stopDemo(); scramble({ random: true }); });
  $("#btn-reset").addEventListener("click", () => { stopDemo(); setPuzzle(SOLVED.slice(), { label: "solved cube", moves: [] }); });
  $("#btn-notation").addEventListener("click", () => {
    const f = $("#notation-form");
    f.hidden = !f.hidden;
    if (!f.hidden) $("#notation-input").focus();
  });
  $("#notation-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const txt = $("#notation-input").value.trim();
    if (txt) scramble({ moves: txt });
  });
  $("#btn-solve").addEventListener("click", () => { stopDemo(); solve(); });
  $("#btn-demo").addEventListener("click", () => (DEMO.on ? stopDemo() : demoLoop()));
  $("#btn-paint").addEventListener("click", () => { stopDemo(); startPaint(); });
  $("#btn-help").addEventListener("click", toggleHelp);
  $("#paint-done").addEventListener("click", () => endPaint(true));
  $("#paint-cancel").addEventListener("click", () => endPaint(false));
  $("#paint-clear").addEventListener("click", () => {
    PAINT.colors = PAINT.colors.map((c, i) => (i % 9 === 4 ? c : 6));
    S.cube.setColors(PAINT.colors);
    renderSwatches();
  });
  $("#tp-play").addEventListener("click", () => (S.playing ? pause() : play()));
  $("#tp-fwd").addEventListener("click", () => { pause(); stepFwd(); });
  $("#tp-back").addEventListener("click", () => { pause(); stepBack(); });
  $("#tp-start").addEventListener("click", () => { pause(); jumpTo(0); });
  $("#tp-end").addEventListener("click", () => { pause(); jumpTo(S.trace ? S.trace.moves.length : 0); });
  $("#attn-toggle").addEventListener("change", renderMind);
  $("#arrow-toggle").addEventListener("change", renderMind);
  $("#coach").addEventListener("change", refreshHint);
  $$("#mem-seg button").forEach((b) => b.addEventListener("click", () => {
    $$("#mem-seg button").forEach((x) => x.setAttribute("aria-checked", x === b));
    S.memMode = b.dataset.mem;
    renderMind();
  }));
  $("#agent-chip").addEventListener("click", () => setSnapshot(null));
  $("#rp-depth").addEventListener("input", () => ($("#rp-depth-out").textContent = $("#rp-depth").value));
  $("#rp-new").addEventListener("click", () => replay());
  $("#theme-btn").addEventListener("click", cycleTheme);
  let rt;
  window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(rerenderAll, 120); });
  window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", rerenderAll);
  document.addEventListener("keydown", onKey);
}

function onKey(e) {
  if ((e.target.closest && e.target.closest("input, select, textarea")) || e.ctrlKey || e.metaKey || e.altKey) return;
  if (location.hash && location.hash !== "#solve") return;
  if (PAINT.on) {
    if (e.key >= "1" && e.key <= "6") { PAINT.sel = +e.key - 1; renderSwatches(); }
    else if (e.key === "Escape") endPaint(false);
    else if (e.key === "Enter") endPaint(true);
    return;
  }
  const k = e.key.toUpperCase();
  const f = S.meta.faces.indexOf(k);
  if (f >= 0 && k.length === 1) { e.preventDefault(); stopDemo(); manualMove(2 * f + (e.shiftKey ? 1 : 0)); return; }
  if (e.key === " ") { e.preventDefault(); S.playing ? pause() : play(); }
  else if (e.key === "ArrowRight") { e.preventDefault(); pause(); stepFwd(); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); pause(); stepBack(); }
  else if (e.key === "Home") { pause(); jumpTo(0); }
  else if (e.key === "End") { pause(); jumpTo(S.trace ? S.trace.moves.length : 0); }
  else if (k === "S") { stopDemo(); solve(); }
  else if (k === "X") { stopDemo(); scramble({ depth: +$("#depth").value }); }
  else if (e.key === "?") toggleHelp();
}

function toggleHelp() {
  let h = $("#help");
  if (h) { h.remove(); return; }
  h = document.createElement("div");
  h.id = "help";
  h.className = "help-overlay";
  h.innerHTML = `<div class="panel help-card"><h2>Keyboard</h2><table>
    <tr><td><kbd>U</kbd> <kbd>R</kbd> <kbd>F</kbd> <kbd>D</kbd> <kbd>L</kbd> <kbd>B</kbd></td><td>turn a face clockwise (<kbd>Shift</kbd>: counter-clockwise)</td></tr>
    <tr><td><kbd>X</kbd></td><td>scramble with the chosen depth</td></tr>
    <tr><td><kbd>S</kbd></td><td>solve</td></tr>
    <tr><td><kbd>Space</kbd></td><td>play / pause the solution</td></tr>
    <tr><td><kbd>←</kbd> <kbd>→</kbd></td><td>step back / forward</td></tr>
    <tr><td><kbd>Home</kbd> <kbd>End</kbd></td><td>jump to the start / end</td></tr>
    <tr><td><kbd>1</kbd>–<kbd>6</kbd>, <kbd>Enter</kbd>, <kbd>Esc</kbd></td><td>paint mode: pick a colour, use the cube, cancel</td></tr>
    <tr><td><kbd>?</kbd></td><td>this help</td></tr></table>
    <p class="sub">Drag the background to look around the cube; scroll to zoom.</p></div>`;
  h.addEventListener("click", () => h.remove());
  document.body.appendChild(h);
}

function cycleTheme() {
  const cur = document.documentElement.dataset.theme || "auto";
  const next = cur === "auto" ? "light" : cur === "light" ? "dark" : "auto";
  if (next === "auto") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = next;
  try { localStorage.setItem("theme", next); } catch (e) {}
  toast(`theme: ${next}`);
  rerenderAll();
}

function rerenderAll() {
  renderMind();
  if (S.training && location.hash === "#training") renderTraining();
  if (location.hash === "#results") renderResults();
}

async function interrupt() {
  S.token++;
  S.playId++;
  S.playing = false;
  updatePlayBtn();
  await S.cube.queue;
  if (!PAINT.on) S.cube.setState(currentState());
}

async function setPuzzle(state, setup) {
  if (PAINT.on) finishPaint();
  await interrupt();
  S.gen++;
  S.start = state.slice();
  S.setup = setup;
  S.trace = null;
  S.pos = 0;
  S.cube.setState(S.start);
  S.cube.showArrow(null);
  hideBanner();
  renderTimeline();
  setStatus();
  renderMind();
  refreshHint();
  $("#solve-meta").innerHTML = "";
}

function clearTrace() {
  if (!S.trace) return;
  const st = currentState();
  const done = S.trace.moves.slice(0, S.pos);
  S.start = st;
  S.setup = { label: S.setup.label, moves: S.setup.moves ? S.setup.moves.concat(done) : null };
  S.trace = null;
  S.pos = 0;
  S.gen++;
  renderTimeline();
  $("#solve-meta").innerHTML = "";
}

async function scramble(body) {
  try {
    const d = await api("/api/scramble", body);
    const label = body.random ? "random cube" : body.moves ? "your moves" : `${d.moves.length}-move scramble`;
    await setPuzzle(SOLVED.slice(), { label, moves: [] });
    S.cube.stopIdle();
    if (d.moves && d.moves.length) {
      const t = S.token, ms = Math.max(45, Math.min(110, 1800 / d.moves.length));
      for (const a of d.moves) {
        if (t !== S.token) return;
        await S.cube.turn(a, ms);
        if (t !== S.token) return;
        S.start = applyMoves(S.start, [a]);
        S.setup.moves.push(a);
        renderTimeline();
      }
    } else {
      S.start = d.state;
      S.setup.moves = null;
      S.cube.setState(d.state);
    }
    renderTimeline();
    setStatus();
    refreshHint();
  } catch (e) { toast(e.message, true); }
}

async function manualMove(a) {
  if (S.busy || PAINT.on || S.stepping) return;
  await interrupt();
  clearTrace();
  S.gen++;
  S.cube.stopIdle();
  S.cube.showArrow(null);
  hideBanner();
  await S.cube.turn(a, speed()[0] * 0.7);
  S.start = applyMoves(S.start, [a]);
  if (S.setup.moves) S.setup.moves.push(a);
  if (S.setup.label === "solved cube") S.setup.label = "your moves";
  renderTimeline();
  setStatus();
  if (isSolved(S.start) && S.setup.moves && S.setup.moves.length) {
    S.cube.celebrate();
    banner("You solved it!", "good");
  }
  refreshHint();
}

async function solve() {
  if (S.busy) return;
  if (PAINT.on) return toast("Finish painting first.");
  await interrupt();
  clearTrace();
  const st = S.start, gen = S.gen;
  if (isSolved(st)) return toast("Scramble the cube first.");
  S.busy = true;
  $("#btn-solve").disabled = true;
  S.cube.stopIdle();
  hideBanner();
  setStatus("busy", "Thinking…");
  try {
    const d = await api("/api/solve", withSnapshot({ state: st, attention: true }));
    if (gen !== S.gen) return;
    S.trace = d;
    S.pos = 0;
    S.cube.setState(currentState());
    setStatus();
    renderSolveMeta(d);
    renderTimeline();
    renderMind();
    if (!d.moves.length) {
      setStatus("failed", "No solution found");
      banner("The agent made no move", "bad");
    } else {
      play();
    }
  } catch (e) {
    if (gen === S.gen) { toast(e.message, true); setStatus(); }
  } finally {
    S.busy = false;
    $("#btn-solve").disabled = false;
  }
}

function renderSolveMeta(d) {
  const n = d.moves.length;
  $("#solve-meta").innerHTML = `<div class="stats">
      <div><b>${d.solved ? n : "✗"}</b><span>moves</span></div>
      <div><b>${d.secs.toFixed(1)} s</b><span>thinking</span></div>
      <div title="The shortest possible solution (proven up to 12 moves)"><b id="opt-cell">…</b><span>shortest</span></div>
    </div>` + (n ? `<div class="solution"><code>${esc(mergeHalves(d.moves))}</code><button class="btn small ghost" id="copy-sol" type="button">Copy</button></div>` : "");
  const cp = $("#copy-sol");
  if (cp) cp.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(mergeHalves(d.moves)); toast("Copied"); }
    catch (e) { toast("Copy failed", true); }
  });
  api("/api/optimal", { state: S.start }).then((o) => {
    const cell = $("#opt-cell");
    if (S.trace === d && cell) cell.textContent = o.distance == null ? `> ${o.limit}` : o.distance;
  }).catch(() => { const c = $("#opt-cell"); if (c) c.textContent = "—"; });
}

function mergeHalves(moves) {
  const out = [];
  for (let i = 0; i < moves.length; i++) {
    if (i + 1 < moves.length && moves[i] === moves[i + 1]) { out.push(S.meta.faces[moves[i] >> 1] + "2"); i++; }
    else out.push(S.meta.move_names[moves[i]]);
  }
  return out.join(" ");
}

function updatePlayBtn() {
  const b = $("#tp-play");
  b.textContent = S.playing ? "❚❚" : "▶";
  b.setAttribute("aria-label", S.playing ? "Pause" : "Play");
}

async function play() {
  if (!S.trace || !S.trace.moves.length) return;
  if (S.pos >= S.trace.moves.length) await jumpTo(0);
  const id = ++S.playId, t = S.token;
  S.playing = true;
  updatePlayBtn();
  while (id === S.playId && t === S.token && S.trace && S.pos < S.trace.moves.length) {
    if (S.stepping) { await S.cube.queue; continue; }
    await stepFwd(true, id);
  }
  if (id === S.playId) { S.playing = false; updatePlayBtn(); }
}

function pause() { S.playId++; S.playing = false; updatePlayBtn(); }

async function stepFwd(fromPlay = false, id = null) {
  if (!S.trace || S.pos >= S.trace.moves.length || S.stepping) return;
  const t = S.token, [turnMs, thinkMs] = speed();
  renderMind();
  if (fromPlay) {
    await sleep(thinkMs);
    if (id !== S.playId || t !== S.token) return;
  }
  if (S.stepping || !S.trace || S.pos >= S.trace.moves.length) return;
  S.stepping = true;
  const i = S.pos;
  try { await S.cube.turn(S.trace.moves[i], turnMs); } finally { S.stepping = false; }
  if (t !== S.token) return;
  S.pos = i + 1;
  afterMove();
}

async function stepBack() {
  if (!S.trace || S.pos <= 0 || S.stepping) return;
  const t = S.token, i = S.pos - 1;
  S.stepping = true;
  try { await S.cube.turn(S.trace.moves[i] ^ 1, speed()[0]); } finally { S.stepping = false; }
  if (t !== S.token) return;
  S.pos = i;
  afterMove();
}

async function jumpTo(k) {
  await interrupt();
  if (!S.trace) return;
  S.pos = Math.max(0, Math.min(k, S.trace.moves.length));
  S.cube.setState(currentState());
  afterMove();
}

function afterMove() {
  renderTimeline();
  renderMind();
  setStatus();
  const n = S.trace ? S.trace.moves.length : 0;
  if (S.trace && S.pos === n && n > 0) {
    if (S.trace.solved) {
      S.cube.showArrow(null);
      S.cube.celebrate();
      banner(`Solved in ${n} moves`, "good");
    } else banner(`Gave up after ${n} moves`, "bad");
  } else hideBanner();
}

function setStatus(kind, text) {
  const pill = $("#status-pill");
  if (!kind) {
    const st = currentState();
    if (isSolved(st)) { kind = "solved"; text = "Solved"; }
    else if (S.trace) {
      kind = "scrambled";
      text = `Step ${S.pos} / ${S.trace.moves.length}`;
    } else { kind = "scrambled"; text = S.setup.label === "random cube" ? "Random cube" : `Scrambled · ${S.setup.label}`; }
  }
  pill.className = "status-pill " + kind;
  $("#status-text").textContent = text;
  $("#tp-count").textContent = S.trace ? `${S.pos} / ${S.trace.moves.length}` : "0 / 0";
}

function banner(text, kind) {
  const b = $("#banner");
  b.textContent = text;
  b.className = "stage-banner " + kind;
  b.hidden = false;
}
function hideBanner() { $("#banner").hidden = true; }

const DEMO = { on: false, gen: 0 };

function stopDemo() {
  if (!DEMO.on) return;
  DEMO.on = false;
  DEMO.gen++;
  $("#btn-demo").classList.remove("on");
  $("#btn-demo").textContent = "▶ Demo loop";
}

async function demoLoop() {
  DEMO.on = true;
  const g = ++DEMO.gen;
  const live = () => DEMO.on && g === DEMO.gen;
  $("#btn-demo").classList.add("on");
  $("#btn-demo").textContent = "■ Stop demo";
  while (live()) {
    const random = Math.random() < 0.4;
    const depth = 4 + Math.floor(Math.random() * 22);
    await scramble(random ? { random: true } : { depth });
    if (!live()) break;
    await sleep(900);
    if (!live()) break;
    await solve();
    while (live() && (S.busy || S.playing)) await sleep(250);
    await sleep(2800);
  }
}

const PAINT = { on: false, colors: null, sel: 0, backup: null };

async function startPaint() {
  if (S.busy || PAINT.on) return;
  await interrupt();
  clearTrace();
  PAINT.on = true;
  PAINT.backup = S.start.slice();
  PAINT.colors = S.start.map((id) => (id / 9) | 0);
  PAINT.sel = 0;
  S.cube.stopIdle();
  S.cube.showArrow(null);
  S.cube.setAttention(null);
  hideBanner();
  S.cube.setColors(PAINT.colors);
  S.cube.setPick(paintAt);
  renderSwatches();
  $("#paint-bar").hidden = false;
  setStatus("busy", "Paint mode: click stickers");
}

function paintAt(i) {
  if (i % 9 === 4) return toast("Centres are fixed: they decide which colour belongs to which face.");
  PAINT.colors[i] = PAINT.sel;
  S.cube.setColors(PAINT.colors);
  renderSwatches();
}

function renderSwatches() {
  const box = $("#swatches");
  box.innerHTML = "";
  S.meta.face_hex.forEach((hex, f) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "swatch";
    b.style.background = hex;
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", f === PAINT.sel);
    const count = PAINT.colors.filter((c) => c === f).length;
    b.title = `${S.meta.color_names[f]}: ${count} of 9 stickers (key ${f + 1})`;
    b.innerHTML = `<span class="n" style="color:${f === 5 ? "rgba(255,255,255,.75)" : ""}">${count}</span>`;
    b.addEventListener("click", () => { PAINT.sel = f; renderSwatches(); });
    box.appendChild(b);
  });
}

async function endPaint(use) {
  if (!PAINT.on) return;
  if (!use) {
    finishPaint();
    S.cube.setState(PAINT.backup);
    setStatus();
    return;
  }
  const grey = PAINT.colors.filter((c) => c === 6).length;
  if (grey) return toast(`${grey} sticker${grey > 1 ? "s are" : " is"} still grey: paint every sticker.`, true);
  try {
    const d = await api("/api/facelets", { facelets: PAINT.colors.map((f) => S.meta.faces[f]).join("") });
    finishPaint();
    await setPuzzle(d.state, { label: "your cube", moves: null });
    toast("Valid cube. Press Solve.");
  } catch (e) { toast("Not a solvable cube: " + e.message, true); }
}

function finishPaint() {
  PAINT.on = false;
  S.cube.setPick(null);
  $("#paint-bar").hidden = true;
}

function chip(a, cls, title) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "mv " + cls;
  b.textContent = S.meta.move_names[a];
  b.style.setProperty("--fc", S.meta.face_hex[a >> 1]);
  if (title) b.title = title;
  return b;
}

function renderTimeline() {
  const tl = $("#timeline");
  tl.innerHTML = "";
  const setup = S.setup.moves;
  if (setup === null) {
    const s = document.createElement("span");
    s.className = "empty";
    s.textContent = S.setup.label === "random cube" ? "random cube (no scramble sequence)" : S.setup.label;
    tl.appendChild(s);
  } else if (setup.length) {
    setup.forEach((a) => tl.appendChild(chip(a, "scr", "scramble")));
  }
  if (S.trace) {
    const sep = document.createElement("span");
    sep.className = "tl-sep";
    sep.textContent = "→ solve";
    tl.appendChild(sep);
    S.trace.moves.forEach((a, i) => {
      const c = chip(a, i < S.pos ? "done" : i === S.pos ? "cur" : "", `move ${i + 1}: jump here`);
      c.addEventListener("click", () => { pause(); jumpTo(i); });
      tl.appendChild(c);
    });
    const cur = tl.querySelector(".cur");
    if (cur) cur.scrollIntoView({ block: "nearest", inline: "nearest" });
  } else if (setup !== null && !setup.length) {
    const s = document.createElement("span");
    s.className = "empty";
    s.textContent = "Scramble the cube or turn a face, then press Solve.";
    tl.appendChild(s);
  }
}

let hintSeq = 0;
async function refreshHint() {
  S.hint = null;
  if (!$("#coach").checked || S.trace || isSolved(S.start)) { renderMind(); return; }
  const seq = ++hintSeq;
  try {
    const d = await api("/api/hint", withSnapshot({ state: S.start }));
    if (seq !== hintSeq) return;
    S.hint = d.step;
    S.hintVersion = d.version;
  } catch (e) { toast(e.message, true); }
  renderMind();
}

function currentStep() {
  if (S.trace && S.pos < S.trace.steps.length) return S.trace.steps[S.pos];
  if (!S.trace && S.hint) return S.hint;
  return null;
}

function currentView() {
  const step = currentStep();
  if (step) return step;
  if (S.trace && S.pos === S.trace.moves.length && S.trace.final) return S.trace.final;
  return null;
}

function renderMind() {
  if (!S.meta || $("#view-solve").hidden) return;
  const step = currentStep();
  const solvedNow = isSolved(currentState());
  $("#mind-sub").textContent = S.trace && step ? `move ${S.pos + 1} / ${S.trace.moves.length}` : S.hint ? "coach" : "";
  const view = currentView();
  const chosen = step ? step.move : -1;
  Ch.drawProbs($("#probs-canvas"), step ? step.probs : null, chosen, S.meta, !step && solvedNow ? "Solved" : null);
  $("#pick").textContent = step ? `${S.meta.move_names[chosen]}  ${Math.round(step.probs[chosen] * 100)} %` : "";
  if (view && !solvedNow) {
    $("#togo").textContent = view.togo == null || view.togo > TOGO_CAP ? `${TOGO_CAP}+` : view.togo.toFixed(1);
    $("#togo-sub").textContent = S.trace && S.trace.solved ? `actually ${S.trace.moves.length - S.pos} left` : "";
  } else {
    $("#togo").textContent = solvedNow ? "0" : "—";
    $("#togo-sub").textContent = "";
  }
  $("#conf").textContent = step ? Math.round(100 * Math.max(0, 1 - step.entropy / LN11)) + " %" : "—";
  const steps = S.trace ? S.trace.steps.concat(S.trace.final ? [S.trace.final] : []) : S.hint ? [S.hint] : [];
  const est = steps.map((s) => (s.togo == null ? TOGO_CAP : Math.min(s.togo, TOGO_CAP)));
  if (S.trace && S.trace.solved && est.length) est[est.length - 1] = 0;
  let real = [];
  if (S.trace && S.trace.solved) {
    const n = S.trace.moves.length;
    real = Array.from({ length: n + 1 }, (_, i) => n - i);
  }
  Ch.drawSpark($("#togo-canvas"), est, real, S.trace ? S.pos : 0);
  let vec = null, lim = 1;
  if (view) {
    if (S.memMode === "h") vec = view.h;
    else if (S.memMode === "c") { vec = view.c; lim = 2; }
    else {
      const prev = S.trace && S.pos > 0 ? S.trace.steps[S.pos - 1].h : null;
      vec = prev ? view.h.map((v, i) => v - prev[i]) : view.h.map(() => 0);
      lim = 0.5;
    }
  }
  Ch.drawMemory($("#mem-canvas"), vec, lim);
  const ver = S.trace ? S.trace.version : S.hint ? S.hintVersion : null;
  ensureMap(ver);
  const sameBasis = S.map && ver && S.map.version === ver;
  const traj = sameBasis ? steps.filter((s) => s.xy).map((s) => s.xy) : [];
  Ch.drawMap($("#map-canvas"), S.map, traj, S.trace ? Math.min(S.pos, traj.length - 1) : traj.length - 1);
  const arrows = $("#arrow-toggle").checked;
  const show = step && (S.trace ? true : $("#coach").checked);
  S.cube.showArrow(show && arrows ? chosen : null);
  S.cube.setAttention($("#attn-toggle").checked && step && step.attn ? step.attn : null);
}

async function loadTraining() {
  try {
    const d = await api("/api/training");
    S.training = d.available ? d : null;
    $("#train-grid").hidden = !d.available;
    if (!d.available) { $("#train-tiles").innerHTML = tile("—", "no training log in this folder"); return; }
    renderTraining();
    renderAgentChip();
    if (!S.replay) replay();
  } catch (e) { toast(e.message, true); }
}

function startPolling() {
  stopPolling();
  S.pollTimer = setInterval(() => { if (location.hash === "#training" && S.training && S.training.live) loadTraining(); }, 8000);
}
function stopPolling() { if (S.pollTimer) clearInterval(S.pollTimer); S.pollTimer = null; }

const SERIES_COLORS = ["#3cd3ff", "#8f7dff", "#ff5ec4", "#ffb347", "#36d987", "#ff6b6b"];

function setLegend(canvas, series) {
  let el = canvas.nextElementSibling;
  if (!el || !el.classList.contains("legend")) {
    el = document.createElement("div");
    el.className = "legend";
    canvas.after(el);
  }
  el.innerHTML = series.map((s) => `<span><i class="${s.dash ? "dash" : ""}" style="${s.dash ? `border-color:${s.color}` : `background:${s.color}`}"></i>${esc(s.name)}</span>`).join("");
}

const tile = (value, label, hint = "") => `<div class="tile" title="${esc(hint)}"><b>${esc(value)}</b><span>${esc(label)}</span></div>`;

function renderTraining() {
  const d = S.training;
  if (!d) return;
  $("#train-tiles").innerHTML = [
    tile(fmtMovesM(d.samples / 1e6), "moves of practice"),
    tile(d.n_envs.toLocaleString(), "cubes at once"),
    tile("−1", "reward per move"),
    tile(`${d.depth} / ${d.max_depth}`, "scramble depth reached", "the hardest scramble the training has reached, in moves"),
  ].join("");
  const later = d.stages.filter((s) => s.from_samples > 0);
  $("#stage-bar").hidden = !later.length;
  const step = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000].find((x) => d.samples / 1e6 / x <= 80) * 1e6;
  const cols = Math.ceil(d.samples / step);
  $("#stage-bar").innerHTML = d.stages.map((s, i) => {
    const end = i + 1 < d.stages.length ? d.stages[i + 1].from_samples : cols * step;
    return `<div style="flex:${end - s.from_samples}" title="${esc(s.hint)}"><b>Stage ${i + 1}</b><span>${esc(s.label)}</span></div>`;
  }).join("");
  const marks = later.map((s, i) => ({ x: s.from_samples / 1e6, label: `Stage ${i + 2}` }));
  const ev = d.evals;
  const mv = (e) => e.samples / 1e6;
  const line = (id, opt) => Ch.lineChart($(id), { xName: "after", xFmt: (v) => fmtMovesM(v) + " moves", xTick: fmtMovesM, marks, ...opt });
  const rows = Array.from({ length: d.max_depth }, (_, i) => String(i + 1)).concat(["random"]);
  const at = Array.from({ length: cols }, (_, i) => ev.filter((e) => e.samples <= i * step).pop() || ev[0]);
  Ch.heatmap($("#heat-canvas"), {
    values: rows.map((k) => at.map((e) => (e && e.results[k] ? e.results[k].solved : null))),
    rowLabels: rows.map((r) => (r === "random" ? "rand" : r)), colLabels: at.map((_, i) => fmtMovesM((i * step) / 1e6)),
    rowName: "depth", colName: "after", valueName: "solved", rowTitle: "scramble depth",
    colMarks: later.map((s, i) => ({ col: s.from_samples / step, label: `Stage ${i + 2}` })),
  });
  const groups = ["5", "10", "15", "20", "random"];
  const series = (key, scale, fmt) => groups.map((k, i) => ({
    name: k === "random" ? "random cube" : "depth " + k, color: SERIES_COLORS[i], fmt,
    pts: ev.map((e) => [mv(e), e.results[k] && e.results[k][key] != null ? e.results[k][key] * scale : null]),
  }));
  const solved = series("solved", 100, (v) => (v == null ? "—" : v.toFixed(0) + " %"));
  line("#ch-solve", { series: solved, leftMin: 0, leftMax: 100, leftFmt: (v) => v + "%" });
  setLegend($("#ch-solve"), solved);
  const lens = series("mean_len", 1, (v) => (v == null ? "—" : v.toFixed(0)));
  line("#ch-len", { series: lens, leftMin: 0 });
  setLegend($("#ch-len"), lens);
  const one = (id, name, col, color, opt = {}) => line(id, { series: [{ name, color, pts: d.log.map((r) => [r[0] / 1e6, r[col]]) }], leftMin: 0, ...opt });
  one("#ch-depth", "scramble depth", 1, SERIES_COLORS[1]);
  one("#ch-ent", "entropy", 2, SERIES_COLORS[2]);
  one("#ch-ev", "explained variance", 3, SERIES_COLORS[4], { leftMin: undefined, leftMax: 1 });
}

async function replay() {
  const strip = $("#replay-strip");
  strip.innerHTML = `<span class="sub">…</span>`;
  let d;
  try { d = await api("/api/replay", { depth: +$("#rp-depth").value, seed: Math.floor(Math.random() * 1e9) }); }
  catch (e) { strip.innerHTML = `<span class="sub">${esc(e.message)}</span>`; return; }
  S.replay = d;
  const maxLen = Math.max(1, ...d.results.filter((r) => r.solved).map((r) => r.length));
  strip.innerHTML = "";
  for (const r of d.results) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "rp-cell " + (r.solved ? "ok" : "no");
    b.title = "Watch this agent try the cube";
    b.innerHTML = `<span class="it">${r.it === 0 ? "untrained" : fmtMovesM((r.samples || 0) / 1e6)}</span>
      <span class="res">${r.solved ? "✓ " + r.length : "✗"}</span><span class="bar">${r.solved ? `<i style="height:${Math.max(12, (100 * r.length) / maxLen)}%"></i>` : ""}</span>`;
    b.addEventListener("click", async () => {
      location.hash = "#solve";
      await setSnapshot(r.it);
      await setPuzzle(d.state, { label: `${d.scramble.length}-move scramble`, moves: d.scramble.slice() });
      solve();
    });
    strip.appendChild(b);
  }
}

async function renderResults() {
  const box = $("#bench");
  let d;
  try { d = await api("/api/benchmark"); }
  catch (e) { box.innerHTML = `<div class="panel empty-state">${esc(e.message)}</div>`; return; }
  if (!d.available) {
    box.innerHTML = `<div class="panel empty-state">Run <code>python -m rubik.benchmark</code> first.</div>`;
    return;
  }
  const head = (x) => `<div class="panel-head"><h2>${esc(x.title)}</h2><span class="sub">${esc(x.sub || "")}</span></div>`;
  const charts = d.curves.map((c, i) => `<div class="panel">${head(c)}<canvas class="chart tall" id="bench-ch-${i}"></canvas></div>`).join("");
  let bars = "";
  if (d.bars) {
    const max = Math.max(...d.bars.rows.map((r) => r.value));
    bars = `<div class="panel bars">${head(d.bars)}` + d.bars.rows.map((r) =>
      `<div class="bar-row${r.own ? " own" : ""}"><span class="name">${esc(r.name)}</span><span class="track"><i style="width:${((100 * r.value) / max).toFixed(1)}%"></i></span><b>${r.value.toFixed(1)}</b><span class="note">${esc(r.note)}</span></div>`).join("") + `</div>`;
  }
  box.innerHTML = `<div class="tiles">${d.tiles.map((t) => tile(t.value, t.label, t.hint)).join("")}</div><div class="grid2">${charts}</div>${bars}`;
  d.curves.forEach((c, i) => {
    const canvas = document.getElementById(`bench-ch-${i}`);
    const series = c.series.map((x, j) => ({ name: x.name, color: SERIES_COLORS[j % SERIES_COLORS.length], pts: x.pts, dash: x.dash, fmt: (v) => (v == null ? "—" : c.pct ? v.toFixed(0) + " %" : v.toFixed(1)) }));
    if (series.length > 1) setLegend(canvas, series);
    Ch.lineChart(canvas, {
      series, marks: c.marks, leftMin: c.yMin, leftMax: c.yMax, xName: c.xName, leftFmt: c.pct ? (v) => v + "%" : undefined,
      xFmt: c.xMoves ? (v) => fmtMovesM(v) : (v) => (Number.isInteger(v) ? v : v.toFixed(1)),
    });
  });
}

function route() {
  const v = (location.hash || "#solve").slice(1);
  const views = ["solve", "training", "results", "about"];
  const view = views.includes(v) ? v : "solve";
  views.forEach((x) => ($("#view-" + x).hidden = x !== view));
  $$(".tabs a").forEach((a) => a.classList.toggle("active", a.dataset.view === view));
  Ch.hideTip();
  if (view === "training") { loadTraining(); startPolling(); } else stopPolling();
  if (view === "results") renderResults();
  if (view === "solve") setTimeout(() => { S.cube.resize(); renderMind(); }, 0);
}

boot().catch((e) => { console.error(e); toast("Failed to start: " + e.message, true); });
