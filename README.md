# ARMONI

Plataforma de tutoría inteligente sensible a la atención: una diadema EEG de bajo costo registra la actividad cerebral mientras el estudiante aprende, ARMONI la convierte en indicadores de aprendizaje interpretables y un equipo de agentes de IA ajusta el siguiente paso. El docente ve los mismos indicadores y siempre tiene la última palabra.

Este repositorio contiene el sitio web del proyecto (rediseño 2026), destinado a **armoni.bio**.

## Páginas

- `/` — inicio: el problema, cómo funciona y la propuesta.
- `/tecnologia/` — arquitectura interactiva (JointJS): toca cualquier bloque para ver qué hace.
- `/investigacion/` — estudio piloto y resultados preliminares.
- `/plataforma/` — módulos de la plataforma y hoja de ruta.

## Desarrollo

Requiere Node 22.12 o superior.

```bash
npm install
npm run dev      # servidor local en http://localhost:4321
npm run build    # sitio estático en dist/
```

## Publicación

Pensado para Cloudflare Pages: comando de compilación `npm run build`, carpeta de salida `dist`, dominio `armoni.bio`.

## Tecnología

[Astro](https://astro.build) · [JointJS](https://github.com/clientIO/joint) (MPL‑2.0) para el diagrama de arquitectura · animaciones en canvas 2D propias.
