// Señales simuladas para demos. No son datos de personas reales.
export type Trace = { color: string; y: number; amp: number; freq: number; noise: number; slow?: boolean };

export function drawTraces(canvas: HTMLCanvasElement, traces: Trace[], opts: { speed?: number } = {}) {
  const ctx = canvas.getContext('2d')!;
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let t = 0, run = false, last = 0, seeds = traces.map(() => Math.random() * 1000);
  function fit() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = canvas.getBoundingClientRect();
    canvas.width = Math.round(r.width * dpr); canvas.height = Math.round(r.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  function frame(now: number) {
    if (!run) return;
    if (now - last < 33) { requestAnimationFrame(frame); return; }
    last = now; t += opts.speed ?? 3;
    paint(); requestAnimationFrame(frame);
  }
  function paint() {
    const W = canvas.clientWidth, H = canvas.clientHeight;
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = 'rgba(255,255,255,0.05)'; ctx.lineWidth = 1;
    for (let x = 0; x < W; x += 48) { ctx.beginPath(); ctx.moveTo(x + .5, 0); ctx.lineTo(x + .5, H); ctx.stroke(); }
    traces.forEach((l, i) => {
      ctx.strokeStyle = l.color; ctx.lineWidth = 1.6; ctx.beginPath();
      const base = l.y * H;
      for (let x = 0; x < W; x += 2) {
        const p = x + t + seeds[i];
        const n = l.noise ? (Math.sin(p * 1.7) * Math.sin(p * 0.37 + i) + Math.sin(p * 0.91)) * l.noise : 0;
        const y = base + Math.sin(p * l.freq) * l.amp + Math.sin(p * l.freq * 2.3 + i) * l.amp * .45 + n;
        x ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      }
      ctx.stroke();
    });
  }
  fit(); paint();
  addEventListener('resize', () => { fit(); paint(); });
  if (!reduce) {
    new IntersectionObserver((e) => {
      if (e[0].isIntersecting && !run) { run = true; requestAnimationFrame(frame); }
      else if (!e[0].isIntersecting) run = false;
    }, { threshold: .15 }).observe(canvas);
  }
  return { set(next: Trace[]) { traces = next; seeds = next.map(() => Math.random() * 1000); paint(); }, pause() { run = false; }, play() { if (!run && !reduce) { run = true; requestAnimationFrame(frame); } } };
}
