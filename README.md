# Learning in public

Sahil's growing learning journal. It starts with neural networks and a browser lab inspired by [Karpathy's micrograd lecture](https://www.youtube.com/watch?v=VMj-3S1tku0) and [my Colab notebook](https://colab.research.google.com/drive/1jlF4JoM39M4RcGP04sYwcRpVe1Qud9tA), but the notes support any subject I study next.

## Run locally

Requires Node.js 22 or newer.

```sh
npm install
npm run dev
```

Run `npm test` for expression parser and autograd checks, and `npm run build` for the static site.

## Add a learning note

Create a Markdown file in `src/content/notes/`. Its filename becomes its URL. Start it with:

```yaml
---
title: "What I learned"
summary: "A one sentence overview."
date: 2026-11-04
topic: "Your new topic"
status: "Exploring"
readingTime: "4 min read"
---
```

Then write the note in Markdown. It appears automatically on the notes page and home page. `status` can be `In progress`, `Exploring`, or `Complete`.

## Interactive lab

Visitors can enter a safe arithmetic expression of `x`, adjust the number and size of hidden layers, select a learning rate and training steps, and watch a small autograd based MLP train. The main visual shows every weighted connection. Selecting a neuron reveals its incoming weights and their change in the latest animation frame; the curve and loss show the effect of training. The computation runs entirely in the browser; no user expression is evaluated as JavaScript. The final layer is linear to fit targets outside the tanh range. Training samples are normalized internally and plotted in their original scale.

## Publishing

The GitHub Actions workflow builds and deploys the site to GitHub Pages when pushed to `main`. In the repository's **Settings → Pages**, select **GitHub Actions** as the build and deployment source. The site is configured for the `sahil9001/learning-in-public` repository path.
