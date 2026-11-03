import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const notes = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/notes' }),
  schema: z.object({
    title: z.string(),
    summary: z.string(),
    date: z.coerce.date(),
    topic: z.string(),
    status: z.enum(['In progress', 'Exploring', 'Complete']),
    readingTime: z.string(),
  }),
});

export const collections = { notes };
