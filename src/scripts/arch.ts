// Diagrama interactivo de arquitectura con JointJS (@joint/core, MPL-2.0).
// Bloques seleccionables, panel explicativo, conexiones animadas y recorrido guiado.
import { dia, shapes } from '@joint/core';

type Kind = 'base' | 'nuevo';
type Status = 'Funciona (prototipo)' | 'En desarrollo' | 'Por construir' | 'Diseño listo' | 'Base conceptual';
type Block = { id: string; title: string; kind: Kind; parent?: string; x: number; y: number; w: number; h: number; what: string; gets: string; gives: string; status: Status };

const B: Block[] = [
  { id: 'interfaz', title: 'Interfaz', kind: 'base', x: 330, y: 0, w: 540, h: 64, status: 'Diseño listo',
    what: 'Donde ocurre la clase: el estudiante resuelve lecciones y tareas interactivas, y el docente sigue al grupo en un panel en tiempo real.',
    gets: 'Mensajes de retroalimentación y acciones aprobadas.', gives: 'Respuestas, tiempos y marcas de evento para sincronizar el EEG.' },

  { id: 'experto', title: 'Módulo experto', kind: 'base', x: 0, y: 110, w: 230, h: 360, status: 'Base conceptual',
    what: 'Contiene el conocimiento del tema y las estrategias de solución correctas. Es la referencia con la que se compara lo que hace el estudiante.',
    gets: 'Consultas del gestor de análisis.', gives: 'Soluciones esperadas, pasos válidos y errores típicos.' },
  { id: 'estrategico', parent: 'experto', title: 'Análisis estratégico', kind: 'base', x: 14, y: 150, w: 202, h: 96, status: 'Base conceptual',
    what: 'Razona sobre la estrategia del estudiante: inferencia, deducción y si la tarea es adecuada para su nivel.', gets: 'Pasos de solución del estudiante.', gives: 'Diagnóstico de la estrategia usada.' },
  { id: 'fisica', parent: 'experto', title: 'Dominio: física', kind: 'base', x: 14, y: 258, w: 202, h: 64, status: 'En desarrollo',
    what: 'Primer dominio: cinemática y movimiento rectilíneo uniforme (tiempo, distancia, velocidad).', gets: 'Ejercicio en curso.', gives: 'Solución y errores frecuentes del tema.' },
  { id: 'computacion', parent: 'experto', title: 'Dominio: computación', kind: 'base', x: 14, y: 334, w: 202, h: 64, status: 'Por construir',
    what: 'Segundo dominio previsto: teoría de la computación (autómatas y lenguajes).', gets: 'Ejercicio en curso.', gives: 'Solución y errores frecuentes del tema.' },

  { id: 'analisis', title: 'Gestor de análisis', kind: 'base', x: 260, y: 110, w: 160, h: 360, status: 'En desarrollo',
    what: 'Compara la respuesta del estudiante con el modelo experto y clasifica el error según la taxonomía.', gets: 'Respuestas de la interfaz y referencias del módulo experto.', gives: 'Error clasificado, con su contexto de atención.' },
  { id: 'contexto', parent: 'analisis', title: '+ Contexto de atención', kind: 'nuevo', x: 274, y: 300, w: 132, h: 76, status: 'En desarrollo',
    what: 'Une cada error con el nivel de atención medido en ese momento. Así un error con atención alta se distingue de uno con atención baja.', gets: 'Indicadores en tiempo real de la capa neuro.', gives: 'Etiqueta de atención para cada evento.' },

  { id: 'instruccion', title: 'Modelo de instrucción', kind: 'base', x: 450, y: 110, w: 300, h: 160, status: 'En desarrollo',
    what: 'Describe qué actividades existen y qué tipos de error hay, para que el sistema elija la siguiente actividad con criterio pedagógico.', gets: 'Errores clasificados.', gives: 'Catálogo de actividades y de errores.' },
  { id: 'actividades', parent: 'instruccion', title: 'Modelo de actividades', kind: 'base', x: 470, y: 150, w: 260, h: 46, status: 'En desarrollo',
    what: 'Lecciones, ejercicios, ejemplos resueltos y pausas activas, con su dificultad.', gets: 'Objetivo de aprendizaje.', gives: 'Siguiente actividad posible.' },
  { id: 'taxonomia', parent: 'instruccion', title: 'Taxonomía de errores', kind: 'base', x: 470, y: 208, w: 260, h: 46, status: 'En desarrollo',
    what: 'Clasificación de los errores típicos de cada tema (por ejemplo, confundir distancia con velocidad).', gets: 'Respuesta incorrecta.', gives: 'Tipo de error.' },

  { id: 'estudiante', title: 'Modelo del estudiante', kind: 'base', x: 450, y: 290, w: 300, h: 180, status: 'En desarrollo',
    what: 'La memoria del sistema sobre cada estudiante: qué sabe, cómo prefiere aprender y cómo se comporta su atención.', gets: 'Resultados, preferencias y patrones de la capa neuro.', gives: 'Estado actual del estudiante para los agentes.' },
  { id: 'personal', parent: 'estudiante', title: 'Información personal', kind: 'base', x: 470, y: 326, w: 125, h: 40, status: 'Diseño listo', what: 'Datos mínimos y consentimiento.', gets: 'Registro.', gives: 'Perfil anonimizado.' },
  { id: 'preferencias', parent: 'estudiante', title: 'Preferencias', kind: 'base', x: 605, y: 326, w: 125, h: 40, status: 'Diseño listo', what: 'Ritmo, formato y tipo de ayuda que mejor funcionan.', gets: 'Interacciones.', gives: 'Preferencias de interacción.' },
  { id: 'conocimiento', parent: 'estudiante', title: 'Conocimiento', kind: 'base', x: 470, y: 374, w: 260, h: 34, status: 'En desarrollo', what: 'Nivel de dominio de cada concepto.', gets: 'Resultados de ejercicios.', gives: 'Mapa de dominio.' },
  { id: 'patrones-est', parent: 'estudiante', title: 'Patrones de atención (EEG)', kind: 'nuevo', x: 470, y: 418, w: 260, h: 36, status: 'En desarrollo',
    what: 'Cómo evoluciona la atención de este estudiante en cada concepto y momento de la sesión.', gets: 'Patrones de la capa neuro.', gives: 'Historial de atención por concepto.' },

  { id: 'feedback', title: 'Gestor de retroalimentación con agentes', kind: 'nuevo', x: 780, y: 110, w: 420, h: 360, status: 'Por construir',
    what: 'Decide qué hacer después. Tres agentes de IA trabajan en cadena y una capa de reglas con el docente tiene la última palabra.', gets: 'Estado del estudiante y catálogo de actividades.', gives: 'Acción pedagógica y mensaje para el estudiante.' },
  { id: 'ag-analisis', parent: 'feedback', title: 'Agente de análisis', kind: 'nuevo', x: 800, y: 152, w: 380, h: 56, status: 'Por construir',
    what: 'Lee conocimiento, errores recientes y nivel de atención, y resume la situación del estudiante.', gets: 'Modelo del estudiante.', gives: 'Diagnóstico breve.' },
  { id: 'ag-plan', parent: 'feedback', title: 'Agente planificador', kind: 'nuevo', x: 800, y: 220, w: 380, h: 56, status: 'Por construir',
    what: 'Elige la acción: pista, ejemplo resuelto, cambio de ritmo o de dificultad, repaso o pausa.', gets: 'Diagnóstico y actividades disponibles.', gives: 'Acción pedagógica.' },
  { id: 'ag-msg', parent: 'feedback', title: 'Generación de mensajes', kind: 'nuevo', x: 800, y: 288, w: 380, h: 56, status: 'Por construir',
    what: 'Filtra, prioriza y redacta la retroalimentación en lenguaje claro para el estudiante.', gets: 'Acción elegida.', gives: 'Mensaje final.' },
  { id: 'reglas', parent: 'feedback', title: 'Reglas y docente en el ciclo', kind: 'nuevo', x: 800, y: 356, w: 380, h: 56, status: 'Por construir',
    what: 'Límites explícitos, explicación de cada decisión y aprobación del docente para acciones sensibles. Todo queda en una bitácora auditable.', gets: 'Propuesta de los agentes.', gives: 'Acción aprobada o corregida.' },

  { id: 'neuro', title: 'Capa neuro de ARMONI', kind: 'nuevo', x: 0, y: 520, w: 1200, h: 200, status: 'Funciona (prototipo)',
    what: 'Convierte la señal de una diadema EEG portátil en indicadores que el tutor puede usar. Es el corazón técnico de ARMONI.', gets: 'Señal cruda del dispositivo y marcas de evento.', gives: 'Indicadores en tiempo real y patrones de aprendizaje.' },
  { id: 'eeg', parent: 'neuro', title: 'EEG portátil', kind: 'nuevo', x: 20, y: 570, w: 210, h: 120, status: 'Funciona (prototipo)',
    what: 'NeuroSky MindWave 2 integrado; Muse y Emotiv EPOC X en integración; fNIRS en modo investigación.', gets: 'Actividad cerebral.', gives: 'Señal cruda por canal.' },
  { id: 'adq', parent: 'neuro', title: 'Adquisición', kind: 'nuevo', x: 260, y: 570, w: 210, h: 120, status: 'Funciona (prototipo)',
    what: 'Un adaptador por dispositivo (BrainFlow, LSL) que entrega todo en el mismo formato y sincroniza la señal con los eventos de la lección.', gets: 'Señal cruda y eventos.', gives: 'Señal sincronizada.' },
  { id: 'proc', parent: 'neuro', title: 'Procesamiento', kind: 'nuevo', x: 500, y: 570, w: 210, h: 120, status: 'Funciona (prototipo)',
    what: 'Filtro pasa-banda y de muesca, rechazo de parpadeos y artefactos, potencias por banda con el método de Welch y medida de calidad.', gets: 'Señal sincronizada.', gives: 'Potencias por banda y calidad.' },
  { id: 'ind', parent: 'neuro', title: 'Indicadores', kind: 'nuevo', x: 740, y: 570, w: 210, h: 120, status: 'Funciona (prototipo)',
    what: 'Índice de compromiso (CEI), índice PRI y nivel de atención en tres escalones, con confianza para cada ventana de tiempo.', gets: 'Potencias por banda.', gives: 'Indicadores en tiempo real.' },
  { id: 'pat', parent: 'neuro', title: 'Patrones de aprendizaje', kind: 'nuevo', x: 980, y: 570, w: 200, h: 120, status: 'En desarrollo',
    what: 'Atención a lo largo de la sesión, perfil por concepto, relación con errores y calificaciones, y línea base individual.', gets: 'Indicadores acumulados.', gives: 'Patrones para el modelo del estudiante.' },
];

const L: [string, string, 'base' | 'nuevo', string?][] = [
  ['interfaz', 'analisis', 'base', 'respuestas'], ['feedback', 'interfaz', 'base', 'retroalimentación'],
  ['experto', 'analisis', 'base'], ['analisis', 'instruccion', 'base'], ['analisis', 'estudiante', 'base'],
  ['instruccion', 'feedback', 'base'], ['estudiante', 'feedback', 'base'],
  ['eeg', 'adq', 'nuevo'], ['adq', 'proc', 'nuevo'], ['proc', 'ind', 'nuevo'], ['ind', 'pat', 'nuevo'],
  ['ind', 'contexto', 'nuevo', 'tiempo real'], ['pat', 'patrones-est', 'nuevo', 'patrones'],
];
const TOUR = ['interfaz', 'eeg', 'adq', 'proc', 'ind', 'contexto', 'pat', 'patrones-est', 'ag-analisis', 'ag-plan', 'ag-msg', 'reglas', 'interfaz'];

const C = { bg: '#0B0D12', box: 'rgba(22,25,33,0.92)', boxNew: 'rgba(61,245,166,0.10)', edge: 'rgba(255,255,255,0.16)', edgeNew: '#3DF5A6', text: '#EEF1F6', mute: '#A7AEBB', blue: '#5B8CFF', green: '#3DF5A6', sel: '#4FD1FF' };

export function architecture(host: HTMLElement | null, panel: HTMLElement | null, tourBtn: HTMLButtonElement | null) {
  if (!host || !panel) return;
  // JointJS fija tamaño en línea sobre su elemento: se usa un lienzo interior para que el contenedor siga siendo fluido
  const canvasEl = document.createElement('div'); canvasEl.className = 'arch-canvas'; host.appendChild(canvasEl);
  const graph = new dia.Graph({}, { cellNamespace: shapes });
  const paper = new dia.Paper({
    el: canvasEl, model: graph, cellViewNamespace: shapes, width: canvasEl.clientWidth || 800, height: canvasEl.clientHeight || 600, gridSize: 1,
    interactive: false, async: false, background: { color: 'transparent' },
    defaultConnector: { name: 'rounded', args: { radius: 10 } },
  });

  const els = new Map<string, dia.Element>();
  B.forEach((b) => {
    const isParent = B.some((c) => c.parent === b.id);
    const el = new shapes.standard.Rectangle({ id: b.id, position: { x: b.x, y: b.y }, size: { width: b.w, height: b.h } });
    el.attr({
      root: { cursor: 'pointer', title: b.title },
      body: { fill: b.kind === 'nuevo' ? C.boxNew : C.box, stroke: b.kind === 'nuevo' ? C.edgeNew : C.edge, strokeWidth: 1.2, rx: 12, ry: 12 },
      label: isParent
        ? { text: b.title, fill: b.kind === 'nuevo' ? '#3DF5A6' : C.text, fontSize: 15, fontWeight: 600, fontFamily: 'Inter Tight Variable, Inter, sans-serif', x: 'calc(0.5*w)', y: 14, textAnchor: 'middle', textVerticalAnchor: 'top', textWrap: { width: -24, maxLineCount: 2, ellipsis: true } }
        : { text: b.title, fill: C.text, fontSize: 13, fontFamily: 'Inter Variable, Inter, sans-serif', textWrap: { width: -16, maxLineCount: 3, ellipsis: true } },
    });
    if (b.id === 'neuro') el.attr('label', { x: 20, textAnchor: 'start' });
    el.addTo(graph); els.set(b.id, el);
  });
  B.forEach((b) => { if (b.parent) els.get(b.parent)!.embed(els.get(b.id)!); });

  const links: dia.Link[] = [];
  L.forEach(([s, t, k, label]) => {
    const link = new shapes.standard.Link({ source: { id: s }, target: { id: t } });
    link.attr({ line: { stroke: k === 'nuevo' ? C.green : C.blue, strokeWidth: 1.6, class: 'arch-flow', targetMarker: { type: 'path', d: 'M 9 -5 0 0 9 5 z', fill: k === 'nuevo' ? C.green : C.blue, stroke: 'none' } } });
    if (label) link.labels([{ position: 0.5, attrs: { text: { text: label, fill: C.mute, fontSize: 11, fontFamily: 'IBM Plex Mono, monospace' }, rect: { fill: C.bg, rx: 4, ry: 4 } } }]);
    link.addTo(graph); links.push(link);
  });
  // Rutas: subir desde la capa neuro hacia los modelos
  links.forEach((l) => {
    const s = (l.source() as any).id, t = (l.target() as any).id;
    if (s === 'ind' && t === 'contexto') l.vertices([{ x: 845, y: 500 }, { x: 340, y: 500 }]);
    if (s === 'pat' && t === 'patrones-est') l.vertices([{ x: 1080, y: 490 }, { x: 600, y: 490 }]);
    if (s === 'feedback' && t === 'interfaz') l.vertices([{ x: 990, y: 32 }]);
    if (s === 'interfaz' && t === 'analisis') { l.source({ id: s, anchor: { name: 'bottom', args: { dx: -170 } } }); }
  });

  const fit = () => { const w = Math.max(host.clientWidth, window.innerWidth < 560 ? 880 : 0), h = host.clientHeight; if (w < 40 || h < 40) return; paper.setDimensions(w, h); paper.transformToFitContent({ padding: 12, useModelGeometry: true, horizontalAlign: 'middle', verticalAlign: 'middle', maxScale: 1.4 }); };
  fit(); new ResizeObserver(fit).observe(host);

  const byId = new Map(B.map((b) => [b.id, b]));
  const kindLabel = (k: Kind) => (k === 'nuevo' ? 'Componente nuevo de ARMONI' : 'Arquitectura base del tutor inteligente');
  function select(id: string | null) {
    graph.getElements().forEach((e) => e.attr({ body: { stroke: byId.get(e.id as string)!.kind === 'nuevo' ? C.edgeNew : C.edge, strokeWidth: 1.2 }, root: { opacity: id ? 0.45 : 1 } }));
    links.forEach((l) => l.attr('root/opacity', id ? 0.3 : 1));
    if (!id) { panel.dataset.empty = 'true'; return; }
    const el = els.get(id)!; const b = byId.get(id)!;
    const keep = new Set<string>([id, ...(b.parent ? [b.parent] : []), ...el.getEmbeddedCells().map((c) => c.id as string)]);
    graph.getConnectedLinks(el, { deep: true }).forEach((l) => { l.attr('root/opacity', 1); keep.add((l.source() as any).id); keep.add((l.target() as any).id); });
    keep.forEach((k) => els.get(k)?.attr('root/opacity', 1));
    el.attr({ body: { stroke: C.sel, strokeWidth: 2.4 } });
    panel.dataset.empty = 'false';
    panel.innerHTML = `
      <span class="arch-kind ${b.kind}">${kindLabel(b.kind)}</span>
      <h3>${b.title}</h3>
      <p>${b.what}</p>
      <dl><dt>Recibe</dt><dd>${b.gets}</dd><dt>Entrega</dt><dd>${b.gives}</dd><dt>Estado</dt><dd><span class="arch-status">${b.status}</span></dd></dl>`;
  }
  paper.on('element:pointerclick', (view) => { stopTour(); select(view.model.id as string); });
  paper.on('blank:pointerclick', () => { stopTour(); select(null); });

  let timer: number | undefined, step = 0;
  function stopTour() { if (timer) { clearInterval(timer); timer = undefined; if (tourBtn) tourBtn.textContent = 'Recorrer el flujo ▶'; } }
  tourBtn?.addEventListener('click', () => {
    if (timer) { stopTour(); return; }
    step = 0; select(TOUR[step]); tourBtn.textContent = 'Detener ■';
    timer = window.setInterval(() => { step++; if (step >= TOUR.length) { stopTour(); return; } select(TOUR[step]); }, 2600);
  });
  select('neuro');
}
