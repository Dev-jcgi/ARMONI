// Efectos de texto e interacción de ARMONI (sin dependencias).
const reduce = () => matchMedia('(prefers-reduced-motion: reduce)').matches;
const GLYPHS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZαβγδθ0123456789·/<>+=';

/** Efecto de "descifrado": los caracteres pasan por símbolos aleatorios hasta fijarse. */
export function scrambleTo(el: HTMLElement, text: string, _ms = 900) {
  el.textContent = text; return Promise.resolve();
  // eslint-disable-next-line no-unreachable
  if (reduce()) { el.textContent = text; return Promise.resolve(); }
  return new Promise<void>((done) => {
    const start = performance.now();
    const tick = (now: number) => {
      const p = Math.min(1, (now - start) / ms);
      const fixed = Math.floor(p * text.length);
      let out = text.slice(0, fixed);
      for (let i = fixed; i < text.length; i++) out += text[i] === ' ' ? ' ' : GLYPHS[(Math.random() * GLYPHS.length) | 0];
      el.textContent = out;
      p < 1 ? requestAnimationFrame(tick) : (el.textContent = text, done());
    };
    requestAnimationFrame(tick);
  });
}

/** Titulares con data-scramble: se descifran al entrar en pantalla. */
export function scrambleOnView() {
  const els = document.querySelectorAll<HTMLElement>('[data-scramble]');
  const io = new IntersectionObserver((es) => es.forEach((e) => {
    if (!e.isIntersecting) return;
    const el = e.target as HTMLElement; io.unobserve(el);
    const text = el.dataset.text || el.textContent || '';
    el.dataset.text = text; scrambleTo(el, text, Math.min(1400, 28 * text.length));
  }), { threshold: 0.4 });
  els.forEach((el) => { el.dataset.text = el.textContent || ''; io.observe(el); });
}

/** Palabras rotativas: <span data-rotate="a|b|c">. */
export function rotateWords() {
  document.querySelectorAll<HTMLElement>('[data-rotate]').forEach((el) => {
    const words = (el.dataset.rotate || '').split('|'); let i = 0;
    if (reduce() || words.length < 2) return;
    setInterval(() => { i = (i + 1) % words.length; scrambleTo(el, words[i], 700); }, 3200);
  });
}

/** Cifras que cuentan desde cero: data-count="84" data-decimals="0" data-prefix="" data-suffix=" %". */
export function countUp() {
  const els = document.querySelectorAll<HTMLElement>('[data-count]');
  const io = new IntersectionObserver((es) => es.forEach((e) => {
    if (!e.isIntersecting) return;
    const el = e.target as HTMLElement; io.unobserve(el);
    const to = Number(el.dataset.count), dec = Number(el.dataset.decimals || 0), pre = el.dataset.prefix || '', suf = el.dataset.suffix || '';
    if (reduce()) { el.textContent = pre + to.toFixed(dec) + suf; return; }
    const t0 = performance.now(), dur = 1400;
    const tick = (now: number) => { const p = Math.min(1, (now - t0) / dur), v = to * (1 - Math.pow(1 - p, 3)); el.textContent = pre + v.toFixed(dec) + suf; if (p < 1) requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
  }), { threshold: 0.5 });
  els.forEach((el) => io.observe(el));
}

/** Lectura simulada en vivo (chip de la portada). */
export function liveReadout(el: HTMLElement | null) {
  if (!el) return;
  const lvl = (c: number) => (c > 1 ? 'ALTA' : c > 0.6 ? 'MEDIA' : 'BAJA');
  let cei = 1.1;
  const upd = () => {
    cei = Math.max(0.35, Math.min(1.6, cei + (Math.random() - 0.5) * 0.18));
    const q = 88 + Math.round(Math.random() * 10);
    el.innerHTML = `<i></i><span class="rq">Fp1 · 512 Hz ·&nbsp;</span>CEI&nbsp;<b>${cei.toFixed(2)}</b>&nbsp;· atención&nbsp;<b>${lvl(cei)}</b><span class="rq">&nbsp;· calidad ${q} %</span>`;
  };
  upd(); if (!reduce()) setInterval(upd, 1100);
}

/** Máquina de escribir para el simulador. */
function typeInto(el: HTMLElement, text: string) {
  el.textContent = text; return;
  // eslint-disable-next-line no-unreachable
  if (reduce()) { el.textContent = text; return; }
  el.textContent = ''; let i = 0;
  const id = setInterval(() => { el.textContent = text.slice(0, ++i); if (i >= text.length) clearInterval(id); }, 14);
}

type State = { att: 'alta' | 'media' | 'baja'; ans: 'correcta' | 'incorrecta'; streak: 'primera' | 'repetida' };
const DECISIONS: Record<string, { action: string; why: string; msg: string; teacher: boolean }> = {
  'alta-correcta': { action: 'Avanzar y subir la dificultad', why: 'Respuesta correcta con atención sostenida: el concepto está dominado.', msg: '¡Bien resuelto! Vamos con un caso un poco más difícil.', teacher: false },
  'media-correcta': { action: 'Avanzar al mismo nivel', why: 'Correcta, pero la atención empieza a bajar: conviene no aumentar la carga.', msg: 'Correcto. Sigamos con un ejercicio parecido para afianzarlo.', teacher: false },
  'baja-correcta': { action: 'Pausa activa breve', why: 'Acertó, pero la atención es baja: la siguiente tarea tiene alto riesgo de error.', msg: 'Buen trabajo. Tómate 60 segundos y continuamos.', teacher: false },
  'alta-incorrecta': { action: 'Pista conceptual', why: 'Error con atención alta: el problema es el concepto, no la concentración.', msg: 'Revisa qué magnitud cambia y cuál se mantiene. ¿Qué relación hay entre distancia y tiempo?', teacher: false },
  'media-incorrecta': { action: 'Ejemplo resuelto paso a paso', why: 'Error con atención media: un ejemplo guiado reduce la carga mientras se recupera el foco.', msg: 'Veamos juntos un ejemplo resuelto antes de intentarlo otra vez.', teacher: false },
  'baja-incorrecta': { action: 'Cambio de actividad', why: 'Error con atención baja: repetir el mismo ejercicio no ayudaría.', msg: 'Cambiemos de formato: una actividad visual corta y luego regresamos.', teacher: false },
};

/** Simulador de decisiones del gestor de retroalimentación. */
export function decisionSim(root: HTMLElement | null) {
  if (!root) return;
  const st: State = { att: 'alta', ans: 'incorrecta', streak: 'primera' };
  const out = {
    action: root.querySelector<HTMLElement>('[data-out="action"]')!, why: root.querySelector<HTMLElement>('[data-out="why"]')!,
    msg: root.querySelector<HTMLElement>('[data-out="msg"]')!, teacher: root.querySelector<HTMLElement>('[data-out="teacher"]')!,
    agents: [...root.querySelectorAll<HTMLElement>('[data-agent]')],
  };
  const render = () => {
    const d = DECISIONS[`${st.att}-${st.ans}`];
    const escalate = st.streak === 'repetida' && st.ans === 'incorrecta';
    scrambleTo(out.action, escalate ? 'Avisar al docente' : d.action, 600);
    out.why.textContent = escalate ? 'Tercer error seguido en el mismo concepto: la regla exige revisión humana antes de continuar.' : d.why;
    typeInto(out.msg, escalate ? 'Tu profesor revisará este tema contigo en un momento.' : d.msg);
    out.teacher.textContent = escalate ? 'Requiere aprobación del docente' : 'Acción automática · registrada en la bitácora';
    out.teacher.classList.toggle('warn', escalate);
    out.agents.forEach((a, i) => { a.classList.remove('on'); setTimeout(() => a.classList.add('on'), reduce() ? 0 : 180 * i); });
  };
  root.querySelectorAll<HTMLButtonElement>('[data-set]').forEach((b) => b.addEventListener('click', () => {
    const [k, v] = (b.dataset.set || '').split(':') as [keyof State, string];
    (st as any)[k] = v;
    root.querySelectorAll<HTMLButtonElement>(`[data-set^="${k}:"]`).forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
    render();
  }));
  render();
}
