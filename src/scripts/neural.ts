// Líneas de señal que salen de la cabeza (motivo del logotipo de ARMONI), en canvas 2D.
// Se detiene fuera de pantalla y respeta prefers-reduced-motion.
type Opts = { origin?: { x: number; y: number }; img?: { w: number; h: number }; spread?: number; count?: number; endX?: number };

const COLORS = ['#3DF5A6', '#5B8CFF', '#4FD1FF'];

export function neuralLines(canvas: HTMLCanvasElement, opts: Opts = {}) {
  const ctx = canvas.getContext('2d')!;
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const count = opts.count ?? 11;
  const seeds = Array.from({ length: count }, (_, i) => ({
    phase: Math.random() * Math.PI * 2,
    speed: 0.35 + Math.random() * 0.4,
    start: (i / (count - 1) - 0.5),
    end: (i / (count - 1) - 0.5) * 1.9 + (Math.random() - 0.5) * 0.12,
    color: COLORS[i % COLORS.length],
    hollow: i % 2 === 0,
  }));
  let W = 0, H = 0, ox = 0, oy = 0, scale = 1, run = false, t = 0, last = 0;

  function fit() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = canvas.getBoundingClientRect();
    W = r.width; H = r.height;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (opts.img && opts.origin) {
      // Misma transformación que object-fit: cover para anclar el origen a la nuca de la imagen
      const s = Math.max(W / opts.img.w, H / opts.img.h);
      const dx = (W - opts.img.w * s) / 2, dy = (H - opts.img.h * s) / 2;
      ox = dx + opts.origin.x * opts.img.w * s; oy = dy + opts.origin.y * opts.img.h * s; scale = s * opts.img.h;
    } else { ox = W * (opts.origin?.x ?? 0.5); oy = H * (opts.origin?.y ?? 0.5); scale = H; }
  }

  function paint() {
    ctx.clearRect(0, 0, W, H);
    const spread = (opts.spread ?? 0.22) * scale;
    const endX = W * (opts.endX ?? 0.94);
    ctx.lineCap = 'round';
    for (const s of seeds) {
      const wob = Math.sin(t * s.speed + s.phase);
      const y0 = oy + s.start * spread * 0.55;
      const y1 = oy + s.end * spread * 1.6 + wob * spread * 0.08;
      const c1x = ox + (endX - ox) * 0.3, c1y = y0 + wob * spread * 0.45;
      const c2x = ox + (endX - ox) * 0.62, c2y = y1 - wob * spread * 0.35;
      ctx.strokeStyle = s.color; ctx.globalAlpha = 0.85;
      ctx.lineWidth = Math.max(1.2, scale * 0.0032);
      ctx.shadowColor = s.color; ctx.shadowBlur = 14;
      ctx.beginPath(); ctx.moveTo(ox, y0); ctx.bezierCurveTo(c1x, c1y, c2x, c2y, endX, y1); ctx.stroke();
      // Sensor al final de la línea (cuadro hueco o relleno, como en el logotipo)
      const q = Math.max(7, scale * 0.012);
      ctx.save(); ctx.translate(endX, y1); ctx.rotate(Math.atan2(y1 - c2y, endX - c2x));
      ctx.shadowBlur = 10;
      if (s.hollow) { ctx.lineWidth = Math.max(1.2, scale * 0.0028); ctx.strokeRect(0, -q / 2, q, q); }
      else { ctx.fillStyle = s.color; ctx.fillRect(0, -q / 2, q, q); }
      ctx.restore();
    }
    ctx.globalAlpha = 1; ctx.shadowBlur = 0;
  }

  function frame(now: number) {
    if (!run) return;
    if (now - last > 33) { t += (now - last) / 1000; last = now; paint(); }
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

export function reveal() {
  const els = document.querySelectorAll('.reveal');
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) { els.forEach((e) => e.classList.add('in')); return; }
  const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } }), { threshold: 0.12 });
  els.forEach((e) => io.observe(e));
}
