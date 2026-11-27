// Portada de ARMONI: el perfil del logotipo se completa con un cráneo de vidrio; el cerebro digital
// vive dentro de la cabeza y los electrodos EEG del cuero cabelludo emiten las líneas hacia la derecha.
// Canvas 2D a ~30 fps; se pausa fuera de pantalla y queda estático con prefers-reduced-motion.
import { HEAD_D, HEAD_ASPECT } from './logo-head';

const MINT = '#3DF5A6', BLUE = '#5B8CFF', CYAN = '#4FD1FF';
const PAL = [MINT, BLUE, CYAN];

// Cerebro en coordenadas normalizadas (ancho 1, alto ~0.8), frente a la izquierda
const OUTLINE: [number, number][] = [
  [0.05, 0.47], [0.02, 0.36], [0.05, 0.23], [0.13, 0.12], [0.25, 0.05], [0.39, 0.015], [0.53, 0.01], [0.66, 0.04],
  [0.78, 0.10], [0.88, 0.19], [0.95, 0.31], [0.975, 0.43], [0.95, 0.53], [0.905, 0.58], [0.935, 0.645], [0.91, 0.72],
  [0.82, 0.765], [0.72, 0.75], [0.665, 0.70], [0.64, 0.73], [0.62, 0.79], [0.575, 0.795], [0.565, 0.72],
  [0.50, 0.675], [0.40, 0.665], [0.30, 0.635], [0.22, 0.575], [0.16, 0.565], [0.10, 0.55],
];
const SULCI: [number, number][][] = [
  [[0.18, 0.53], [0.32, 0.47], [0.48, 0.43], [0.62, 0.40], [0.72, 0.36]],
  [[0.52, 0.03], [0.50, 0.13], [0.46, 0.24], [0.44, 0.33], [0.42, 0.42]],
  [[0.60, 0.05], [0.58, 0.16], [0.55, 0.26], [0.53, 0.35]],
  [[0.40, 0.04], [0.37, 0.14], [0.33, 0.25], [0.31, 0.35]],
  [[0.12, 0.20], [0.22, 0.24], [0.30, 0.20], [0.36, 0.25]],
  [[0.08, 0.35], [0.18, 0.37], [0.26, 0.33], [0.34, 0.36]],
  [[0.68, 0.10], [0.70, 0.20], [0.76, 0.27], [0.86, 0.30]],
  [[0.74, 0.40], [0.82, 0.43], [0.90, 0.46]],
  [[0.30, 0.56], [0.42, 0.53], [0.54, 0.55], [0.62, 0.52]],
  [[0.70, 0.63], [0.78, 0.61], [0.88, 0.63]],
  [[0.69, 0.68], [0.78, 0.67], [0.89, 0.69]],
  [[0.71, 0.725], [0.80, 0.72], [0.87, 0.735]],
];
// Cráneo que completa el perfil del logotipo (coordenadas en anchos de la silueta; la coronilla del logo es (0.937, 0))
const SKULL = 'M0.937 0 C1.36 -0.09 1.86 0.13 1.97 0.64 C2.05 1.04 1.86 1.36 1.46 1.5 C1.13 1.61 0.88 1.6 0.74 1.62';
// Posiciones de electrodos a lo largo del cráneo (fracción de la curva)
const ELECTRODES = [0.06, 0.16, 0.27, 0.38, 0.49, 0.6, 0.7];

function smooth(ctx: CanvasRenderingContext2D | Path2D, pts: [number, number][], closed: boolean) {
  const n = pts.length;
  const mid = (a: [number, number], b: [number, number]): [number, number] => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  if (closed) {
    const m0 = mid(pts[n - 1], pts[0]); ctx.moveTo(m0[0], m0[1]);
    for (let i = 0; i < n; i++) { const p = pts[i], m = mid(p, pts[(i + 1) % n]); ctx.quadraticCurveTo(p[0], p[1], m[0], m[1]); }
  } else {
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < n - 1; i++) { const m = mid(pts[i], pts[i + 1]); ctx.quadraticCurveTo(pts[i][0], pts[i][1], m[0], m[1]); }
    ctx.lineTo(pts[n - 1][0], pts[n - 1][1]);
  }
}

// Muestrea la curva del cráneo (3 cúbicas) para colocar electrodos
function skullPoint(u: number): [number, number] {
  const segs: number[][] = [[0.937, 0, 1.36, -0.09, 1.86, 0.13, 1.97, 0.64], [1.97, 0.64, 2.05, 1.04, 1.86, 1.36, 1.46, 1.5], [1.46, 1.5, 1.13, 1.61, 0.88, 1.6, 0.74, 1.62]];
  const s = Math.min(2, Math.floor(u * 3)), t = u * 3 - s, [x0, y0, x1, y1, x2, y2, x3, y3] = segs[s], m = 1 - t;
  return [m * m * m * x0 + 3 * m * m * t * x1 + 3 * m * t * t * x2 + t * t * t * x3, m * m * m * y0 + 3 * m * m * t * y1 + 3 * m * t * t * y2 + t * t * t * y3];
}

export function armoniHero(canvas: HTMLCanvasElement) {
  const ctx = canvas.getContext('2d')!;
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const brainPath = new Path2D(); smooth(brainPath, OUTLINE, true); brainPath.closePath();
  const headPath = new Path2D(HEAD_D), skullPath = new Path2D(SKULL);
  const probe = document.createElement('canvas').getContext('2d')!;

  const nodes: { x: number; y: number; r: number; ph: number }[] = [];
  for (let g = 0; nodes.length < 170 && g < 6000; g++) {
    const x = Math.random(), y = Math.random() * 0.86;
    if (probe.isPointInPath(brainPath, x, y)) nodes.push({ x, y, r: 0.6 + Math.random() * 1.2, ph: Math.random() * 6.28 });
  }
  const edges: [number, number][] = [];
  nodes.forEach((a, i) => {
    nodes.map((b, j) => ({ j, d: (a.x - b.x) ** 2 + (a.y - b.y) ** 2 })).filter((q) => q.j !== i).sort((p, q) => p.d - q.d).slice(0, 3)
      .forEach((q) => { if (q.d < 0.012) edges.push([i, q.j]); });
  });
  const pulses = Array.from({ length: 24 }, () => ({ e: Math.floor(Math.random() * edges.length), t: Math.random(), v: 0.4 + Math.random() * 0.8 }));
  const cables = ELECTRODES.map((u, i) => ({ u, end: i / (ELECTRODES.length - 1) - 0.5, ph: Math.random() * 6.28, sp: 0.35 + Math.random() * 0.35, color: PAL[i % 3], hollow: i % 2 === 0 }));

  let W = 0, H = 0, hx = 0, hy = 0, hw = 0, bx = 0, by = 0, bw = 0, run = false, t = 0, last = 0, mobile = false;
  const tilt = -0.06, cs = Math.cos(tilt), sn = Math.sin(tilt);

  function fit() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = canvas.getBoundingClientRect(); W = r.width; H = r.height;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    mobile = W < 760;
    if (mobile) { hw = Math.min(W * 0.2, 96); hx = W * 0.12; hy = 96; }
    else { hw = Math.min(H * 0.26, W * 0.14); hx = W * 0.5; hy = H * 0.5 - hw * 1.12; }
    bw = hw * 1.45; bx = hx + hw * 1.15; by = hy + hw * 0.62;
  }
  const head = (x: number, y: number): [number, number] => [hx + x * hw, hy + y * hw];

  function drawHead() {
    ctx.save();
    ctx.translate(hx, hy); ctx.scale(hw, hw);
    // Cráneo de vidrio
    ctx.lineWidth = 1.4 / hw; ctx.strokeStyle = 'rgba(79,209,255,0.55)'; ctx.shadowColor = CYAN; ctx.shadowBlur = 16; ctx.stroke(skullPath);
    const glass = ctx.createRadialGradient(1.25, 0.65, 0.1, 1.25, 0.65, 1.0);
    glass.addColorStop(0, 'rgba(91,140,255,0.10)'); glass.addColorStop(1, 'rgba(91,140,255,0)');
    ctx.fillStyle = glass; ctx.beginPath(); ctx.arc(1.25, 0.7, 1.0, 0, 6.29); ctx.fill();
    // Perfil del logotipo
    const g = ctx.createLinearGradient(0, 0, 1, HEAD_ASPECT);
    g.addColorStop(0, '#2B63E0'); g.addColorStop(1, '#13306B');
    ctx.globalAlpha = 0.92; ctx.fillStyle = g; ctx.shadowColor = BLUE; ctx.shadowBlur = 34; ctx.fill(headPath);
    ctx.globalAlpha = 1; ctx.lineWidth = 1.3 / hw; ctx.strokeStyle = 'rgba(79,209,255,0.8)'; ctx.shadowBlur = 14; ctx.stroke(headPath);
    ctx.restore();
  }

  function drawBrain() {
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.translate(bx, by); ctx.rotate(tilt); ctx.translate(-0.5 * bw, -0.42 * bw); ctx.scale(bw, bw);
    const px = 1 / bw;
    ctx.fillStyle = 'rgba(61,245,166,0.05)'; ctx.fill(brainPath);
    ctx.lineWidth = 1.6 * px; ctx.strokeStyle = 'rgba(61,245,166,0.85)'; ctx.shadowColor = MINT; ctx.shadowBlur = 14; ctx.stroke(brainPath);
    ctx.shadowBlur = 6;
    SULCI.forEach((sv, i) => {
      ctx.beginPath(); smooth(ctx, sv.map(([x, y]) => [x, y + Math.sin(t * 0.6 + i) * 0.003] as [number, number]), false);
      ctx.lineWidth = (i < 2 ? 1.5 : 1) * px; ctx.strokeStyle = i < 2 ? 'rgba(79,209,255,0.85)' : i > 8 ? 'rgba(61,245,166,0.7)' : 'rgba(91,140,255,0.6)'; ctx.stroke();
    });
    ctx.shadowBlur = 0; ctx.lineWidth = 0.6 * px; ctx.strokeStyle = 'rgba(91,140,255,0.3)'; ctx.beginPath();
    edges.forEach(([a, b]) => { ctx.moveTo(nodes[a].x, nodes[a].y); ctx.lineTo(nodes[b].x, nodes[b].y); }); ctx.stroke();
    nodes.forEach((n) => { const a = 0.35 + 0.4 * Math.sin(t * 1.4 + n.ph); ctx.fillStyle = `rgba(160,235,255,${a.toFixed(2)})`; ctx.beginPath(); ctx.arc(n.x, n.y, n.r * 1.3 * px, 0, 6.29); ctx.fill(); });
    ctx.shadowColor = MINT; ctx.shadowBlur = 10;
    pulses.forEach((p) => { const [a, b] = edges[p.e]; const x = nodes[a].x + (nodes[b].x - nodes[a].x) * p.t, y = nodes[a].y + (nodes[b].y - nodes[a].y) * p.t; ctx.fillStyle = 'rgba(61,245,166,1)'; ctx.beginPath(); ctx.arc(x, y, 2.2 * px, 0, 6.29); ctx.fill(); });
    ctx.restore();
  }

  function drawCables() {
    const endX = W * (mobile ? 0.97 : 0.96), spread = mobile ? hw * 2.6 : Math.min(H * 0.7, hw * 3.2);
    const midY = hy + hw * 0.75;
    ctx.save(); ctx.globalCompositeOperation = 'lighter'; ctx.lineCap = 'round';
    cables.forEach((c) => {
      const [sx, sy] = skullPoint(c.u); const [x0, y0] = head(sx, sy);
      const wob = Math.sin(t * c.sp + c.ph), y1 = midY + c.end * spread + wob * spread * 0.03;
      const cx1 = x0 + (endX - x0) * 0.35, cx2 = x0 + (endX - x0) * 0.7;
      ctx.strokeStyle = c.color; ctx.globalAlpha = 0.95; ctx.lineWidth = Math.max(1.4, hw * 0.012); ctx.shadowColor = c.color; ctx.shadowBlur = 12;
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.bezierCurveTo(cx1, y0 + wob * hw * 0.15, cx2, y1 - wob * spread * 0.05, endX, y1); ctx.stroke();
      // Electrodo en el cuero cabelludo
      ctx.fillStyle = c.color; ctx.beginPath(); ctx.arc(x0, y0, Math.max(3, hw * 0.035), 0, 6.29); ctx.fill();
      ctx.globalAlpha = 0.35 + 0.35 * Math.sin(t * 2 + c.ph); ctx.beginPath(); ctx.arc(x0, y0, Math.max(6, hw * 0.07), 0, 6.29); ctx.stroke(); ctx.globalAlpha = 0.95;
      // Sensor final
      const q = Math.max(8, hw * 0.05);
      ctx.save(); ctx.translate(endX, y1); ctx.rotate(Math.atan2(wob * spread * 0.05, endX - cx2));
      if (c.hollow) { ctx.lineWidth = Math.max(1.4, hw * 0.01); ctx.strokeRect(0, -q / 2, q, q); } else ctx.fillRect(0, -q / 2, q, q);
      ctx.restore();
    });
    ctx.restore(); ctx.globalAlpha = 1; ctx.shadowBlur = 0;
  }

  function paint() { ctx.clearRect(0, 0, W, H); drawHead(); drawBrain(); drawCables(); }
  function step(dt: number) { t += dt; pulses.forEach((p) => { p.t += p.v * dt; if (p.t > 1) { p.t = 0; p.e = Math.floor(Math.random() * edges.length); } }); }
  function frame(now: number) {
    if (!run) return;
    const dt = (now - last) / 1000;
    if (dt > 0.033) { step(Math.min(dt, 0.1)); last = now; paint(); }
    requestAnimationFrame(frame);
  }
  fit(); paint();
  addEventListener('resize', () => { fit(); paint(); });
  if (reduce) return;
  new IntersectionObserver((e) => {
    if (e[0].isIntersecting && !run) { run = true; last = performance.now(); requestAnimationFrame(frame); }
    else if (!e[0].isIntersecting) run = false;
  }, { threshold: 0.05 }).observe(canvas);
}
