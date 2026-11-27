// @ts-check
import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';

// ARMONI — neurotecnología para investigación educativa — https://armoni.bio (Cloudflare Pages)
export default defineConfig({
  site: 'https://armoni.bio',
  trailingSlash: 'ignore',
  integrations: [sitemap()],
});
