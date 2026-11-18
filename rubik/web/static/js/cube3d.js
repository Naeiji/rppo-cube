import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";
import { RoundedBoxGeometry } from "three/addons/RoundedBoxGeometry.js";
import { RoomEnvironment } from "three/addons/RoomEnvironment.js";

const FACE_NORMALS = [[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, -1, 0], [-1, 0, 0], [0, 0, -1]];

const IDLE_SECONDS = 30;
const SOFTWARE_RENDERER = /llvmpipe|softpipe|lavapipe|swiftshader|swrast|software|basic render/i;

function drawsInSoftware(gl) {
  try {
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    const name = String((ext && gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)) || gl.getParameter(gl.RENDERER) || "");
    return SOFTWARE_RENDERER.test(name);
  } catch (e) {
    return false;
  }
}

function roundedSquare(size, r) {
  const s = size / 2, sh = new THREE.Shape();
  sh.moveTo(-s + r, -s);
  sh.lineTo(s - r, -s); sh.quadraticCurveTo(s, -s, s, -s + r);
  sh.lineTo(s, s - r); sh.quadraticCurveTo(s, s, s - r, s);
  sh.lineTo(-s + r, s); sh.quadraticCurveTo(-s, s, -s, s - r);
  sh.lineTo(-s, -s + r); sh.quadraticCurveTo(-s, -s, -s + r, -s);
  return sh;
}

const easeInOut = (u) => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2);

function shadowTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 256;
  const g = c.getContext("2d");
  const grd = g.createRadialGradient(128, 128, 10, 128, 128, 128);
  grd.addColorStop(0, "rgba(0,0,0,0.55)");
  grd.addColorStop(0.55, "rgba(0,0,0,0.18)");
  grd.addColorStop(1, "rgba(0,0,0,0)");
  g.fillStyle = grd;
  g.fillRect(0, 0, 256, 256);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

export class Cube3D {
  constructor(container, meta) {
    this.container = container;
    this.meta = meta;
    this.state = Array.from({ length: 54 }, (_, i) => i);
    this.queue = Promise.resolve();
    this.animating = 0;
    this.dirty = true;
    this.attn = null;

    const r = (this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" }));
    r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    r.outputColorSpace = THREE.SRGBColorSpace;
    r.toneMapping = THREE.NeutralToneMapping;
    r.toneMappingExposure = 1.0;
    container.appendChild(r.domElement);

    this.scene = new THREE.Scene();
    const pmrem = new THREE.PMREMGenerator(r);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environmentIntensity = 0.55;
    this.camera = new THREE.PerspectiveCamera(30, 1, 0.1, 100);
    this.camera.position.set(5.6, 4.6, 7.2);
    this.controls = new OrbitControls(this.camera, r.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.enablePan = false;
    this.controls.minDistance = 7;
    this.controls.maxDistance = 18;
    this.software = drawsInSoftware(r.getContext());
    this.controls.autoRotate = !this.software;
    this.controls.autoRotateSpeed = 0.55;
    this.idleUntil = performance.now() + IDLE_SECONDS * 1000;
    this.controls.addEventListener("start", () => this.stopIdle());

    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(4, 8, 6);
    const fill = new THREE.DirectionalLight(0xbfd4ff, 0.5);
    fill.position.set(-6, -2, -4);
    this.scene.add(key, fill, new THREE.HemisphereLight(0xdfe8ff, 0x20242c, 0.35));

    this.root = new THREE.Group();
    this.scene.add(this.root);
    const shadow = new THREE.Mesh(new THREE.PlaneGeometry(7, 7), new THREE.MeshBasicMaterial({ map: shadowTexture(), transparent: true, depthWrite: false }));
    shadow.rotation.x = -Math.PI / 2;
    shadow.position.y = -2.35;
    this.scene.add(shadow);

    this._build();
    this._paint();
    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
    const loop = () => {
      requestAnimationFrame(loop);
      if (this.controls.autoRotate && performance.now() > this.idleUntil) this.stopIdle();
      const moved = this.controls.update();
      if (moved || this.dirty || this.animating > 0) {
        r.render(this.scene, this.camera);
        this.dirty = false;
      }
    };
    loop();
  }

  _build() {
    const body = new THREE.MeshPhysicalMaterial({ color: 0x0b0c10, roughness: 0.42, metalness: 0.0, clearcoat: 0.5, clearcoatRoughness: 0.35 });
    const geo = new RoundedBoxGeometry(0.97, 0.97, 0.97, 4, 0.11);
    const stickerGeo = new THREE.ShapeGeometry(roundedSquare(0.82, 0.13), 6);
    const byPos = new Map();
    this.cubies = [];
    for (let x = -1; x <= 1; x++) for (let y = -1; y <= 1; y++) for (let z = -1; z <= 1; z++) {
      if (!x && !y && !z) continue;
      const mesh = new THREE.Mesh(geo, body);
      mesh.position.set(x, y, z);
      this.root.add(mesh);
      const c = { mesh, pos: [x, y, z] };
      this.cubies.push(c);
      byPos.set(`${x},${y},${z}`, c);
    }
    this.stickers = [];
    const zAxis = new THREE.Vector3(0, 0, 1);
    for (let i = 0; i < 54; i++) {
      const p = this.meta.sticker_pos[i], n = this.meta.sticker_normal[i];
      const mat = new THREE.MeshPhysicalMaterial({ color: 0xffffff, roughness: 0.38, metalness: 0.0, clearcoat: 0.6, clearcoatRoughness: 0.25, emissive: 0x000000 });
      const m = new THREE.Mesh(stickerGeo, mat);
      const nv = new THREE.Vector3(...n);
      m.position.copy(nv).multiplyScalar(0.4875);
      m.quaternion.setFromUnitVectors(zAxis, nv);
      m.userData.index = i;
      byPos.get(p.join(",")).mesh.add(m);
      this.stickers.push({ mesh: m, mat });
    }
    this.faceColors = this.meta.face_hex.map((h) => new THREE.Color(h)).concat([new THREE.Color(0x5b6170)]);
    this.raycaster = new THREE.Raycaster();
    const el = this.renderer.domElement;
    let down = null;
    el.addEventListener("pointerdown", (e) => { down = [e.clientX, e.clientY]; });
    el.addEventListener("pointerup", (e) => {
      if (!down || !this.pickCb) return;
      const moved = Math.hypot(e.clientX - down[0], e.clientY - down[1]);
      down = null;
      if (moved > 6) return;
      const r = el.getBoundingClientRect();
      const ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      this.raycaster.setFromCamera(ndc, this.camera);
      const hit = this.raycaster.intersectObjects(this.stickers.map((st) => st.mesh), false)[0];
      if (hit) this.pickCb(hit.object.userData.index);
    });
  }

  setPick(cb) {
    this.pickCb = cb;
    this.renderer.domElement.style.cursor = cb ? "crosshair" : "";
    if (!cb) { this.paintColors = null; this._paint(); }
  }

  setColors(cols) {
    this.paintColors = cols.slice();
    for (let i = 0; i < 54; i++) {
      this.stickers[i].mat.color.copy(this.faceColors[cols[i]]);
      this.stickers[i].mat.emissive.setRGB(0, 0, 0);
    }
    this.dirty = true;
  }

  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.fov = w / h < 1 ? 30 / Math.max(w / h, 0.55) : 30;
    this.camera.updateProjectionMatrix();
    this.dirty = true;
  }

  stopIdle() { this.controls.autoRotate = false; }

  _paint() {
    if (this.paintColors) return this.setColors(this.paintColors);
    const dark = new THREE.Color(0x15171c);
    for (let i = 0; i < 54; i++) {
      const base = this.faceColors[(this.state[i] / 9) | 0];
      const s = this.stickers[i];
      if (this.attn) {
        const a = this.attn[i] ?? 0;
        s.mat.color.copy(base).lerp(dark, 0.78 * (1 - a));
        s.mat.emissive.copy(base).multiplyScalar(0.55 * a * a);
      } else {
        s.mat.color.copy(base);
        s.mat.emissive.setRGB(0, 0, 0);
      }
    }
    this.dirty = true;
  }

  setState(state) {
    this.state = state.slice();
    this.paintColors = null;
    this._paint();
  }

  setAttention(arr) {
    this.attn = arr ? arr.slice() : null;
    this._paint();
  }

  turn(move, ms = 260) {
    if (!Number.isInteger(move) || move < 0 || move > 11) return this.queue;
    this.queue = this.queue.then(() => this._turn(move, ms)).catch((e) => console.error("turn failed", e));
    return this.queue;
  }

  _turn(move, ms) {
    return new Promise((resolve) => {
      const face = move >> 1, cw = (move & 1) === 0;
      const n = FACE_NORMALS[face];
      const axis = new THREE.Vector3(...n);
      const layer = this.cubies.filter((c) => c.pos[0] * n[0] + c.pos[1] * n[1] + c.pos[2] * n[2] === 1);
      const finish = () => {
        try {
          this.state = this.meta.move_src[move].map((i) => this.state[i]);
          this._paint();
        } finally { resolve(); }
      };
      if (ms <= 0) return finish();
      const pivot = new THREE.Group();
      this.root.add(pivot);
      layer.forEach((c) => pivot.attach(c.mesh));
      const target = (cw ? -1 : 1) * Math.PI / 2;
      const t0 = performance.now();
      this.animating++;
      const step = (now) => {
        const u = Math.min(1, (now - t0) / ms);
        pivot.quaternion.setFromAxisAngle(axis, target * easeInOut(u));
        this.dirty = true;
        if (u < 1) return requestAnimationFrame(step);
        try {
          layer.forEach((c) => {
            this.root.attach(c.mesh);
            c.mesh.position.set(...c.pos);
            c.mesh.quaternion.identity();
          });
          this.root.remove(pivot);
        } finally {
          this.animating--;
          finish();
        }
      };
      requestAnimationFrame(step);
    });
  }

  showArrow(move, color = 0x3cd3ff) {
    if (this.arrow) {
      this.scene.remove(this.arrow);
      const geos = new Set();
      this.arrow.traverse((o) => o.geometry && geos.add(o.geometry));
      geos.forEach((g) => g.dispose());
      this.arrow = null;
    }
    if (move == null || move < 0) { this.dirty = true; return; }
    const face = move >> 1, cw = (move & 1) === 0;
    const n = new THREE.Vector3(...FACE_NORMALS[face]);
    const helper = Math.abs(n.y) > 0.9 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(0, 1, 0);
    const u = new THREE.Vector3().crossVectors(helper, n).normalize();
    const v = new THREE.Vector3().crossVectors(n, u);
    const center = n.clone().multiplyScalar(1.66), radius = 1.08, span = Math.PI * 0.72, start = Math.PI * 0.62;
    const dir = cw ? -1 : 1, pts = [];
    for (let k = 0; k <= 40; k++) {
      const th = start + dir * span * (k / 40);
      pts.push(center.clone().addScaledVector(u, radius * Math.cos(th)).addScaledVector(v, radius * Math.sin(th)));
    }
    const solid = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.95, toneMapped: false });
    const ghost = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.28, toneMapped: false, depthTest: false, depthWrite: false });
    const tube = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 60, 0.05, 10);
    const cone = new THREE.ConeGeometry(0.15, 0.34, 20);
    const end = pts[pts.length - 1], tangent = end.clone().sub(pts[pts.length - 3]).normalize();
    const g = new THREE.Group();
    for (const [mat, order] of [[ghost, 10], [solid, 11]]) {
      const t = new THREE.Mesh(tube, mat);
      const head = new THREE.Mesh(cone, mat);
      head.position.copy(end).addScaledVector(tangent, 0.1);
      head.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), tangent);
      t.renderOrder = head.renderOrder = order;
      g.add(t, head);
    }
    this.arrow = g;
    this.scene.add(g);
    this.dirty = true;
  }

  celebrate() {
    const t0 = performance.now();
    this.animating++;
    const step = (now) => {
      const u = Math.min(1, (now - t0) / 900);
      const s = 1 + 0.07 * Math.sin(Math.PI * u) * (1 - u * 0.4);
      this.root.scale.setScalar(s);
      this.root.rotation.y = 0.35 * Math.sin(Math.PI * u) * (1 - u);
      const glow = 0.35 * Math.sin(Math.PI * Math.min(1, u * 1.6));
      this.stickers.forEach((st, i) => st.mat.emissive.copy(this.faceColors[(this.state[i] / 9) | 0]).multiplyScalar(Math.max(glow, 0)));
      this.dirty = true;
      if (u < 1) return requestAnimationFrame(step);
      this.root.scale.setScalar(1);
      this.root.rotation.y = 0;
      this.animating--;
      this._paint();
    };
    requestAnimationFrame(step);
  }
}
