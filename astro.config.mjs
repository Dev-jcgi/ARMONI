// @ts-check
import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';

// ARMONI — neurotecnología para investigación educativa.
// Por defecto https://armoni.bio; en GitHub Pages se compila con SITE_URL=https://jcgi.lat y BASE_PATH=/ARMONI.
export default defineConfig({
  site: process.env.SITE_URL || 'https://armoni.bio',
  base: process.env.BASE_PATH || '/',
  trailingSlash: 'ignore',
  integrations: [sitemap()],
});
