import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://sahil9001.github.io',
  base: process.env.GITHUB_PAGES === 'true' ? '/learning-in-public' : '/',
});
