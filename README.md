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

Cada cambio en `main` se compila y se publica solo en GitHub Pages, en **https://jcgi.lat/ARMONI/** (flujo `.github/workflows/deploy.yml`).

La ruta se controla con dos variables de compilación:

- `SITE_URL`: dominio del sitio (por defecto `https://armoni.bio`).
- `BASE_PATH`: carpeta donde vive (por defecto `/`).

Para usar el dominio **armoni.bio**, apunta su DNS a GitHub Pages, configura el dominio en *Settings → Pages* y cambia en el flujo `SITE_URL` a `https://armoni.bio` y `BASE_PATH` a `/`. En Cloudflare Pages basta con `npm run build` y la carpeta `dist`, sin variables.

## Tecnología

[Astro](https://astro.build) · [JointJS](https://github.com/clientIO/joint) (MPL‑2.0) para el diagrama de arquitectura · animaciones en canvas 2D propias.
