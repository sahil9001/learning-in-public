---
title: "From bigrams to a three-character context"
summary: "Moving from one-character predictions to a small neural network that uses three previous characters to generate names."
date: 2026-11-11
topic: "Language models"
status: "Exploring"
readingTime: "5 min read"
---

After working through micrograd, I wanted to see how a neural network could make a prediction about text. I started with names: given the characters so far, what character is likely to come next?

I put the full experiment in [the notebook on this site](../../notebooks/character-model/). You can also [download the `.ipynb` file](../../notebooks/character_model.ipynb) or [open it in Colab](https://colab.research.google.com/drive/1q_kgY_3X28sdhLr0zyheT8AiXRnyB3-S).

## From a bigram to a wider context

A character bigram model looks at **one** character to predict the next. For example, after `a`, it learns which characters often follow `a`. It cannot tell whether that `a` appears at the start of a name or after `m` and `i`.

This notebook takes the next step: it uses the **previous three characters**. For a name beginning `emma`, the training examples start like this, where `.` is the start/end token:

| Context | Target |
| --- | --- |
| `...` | `e` |
| `..e` | `m` |
| `.em` | `m` |
| `emm` | `a` |
| `mma` | `.` |

The model is a small character-level neural language model, rather than a bigram model. The wider context lets it make different predictions for the same last character depending on what came before it.

<figure class="article-figure">
  <img src="../../images/context-three-prediction.png" alt="Three context characters, dot, e, and m, become embedding vectors that feed a neural network; the model assigns the highest illustrated probability to m as the next character." width="1752" height="897" loading="lazy" />
  <figcaption>A simplified view of the prediction. The notebook uses 10-dimensional embeddings and scores every character in its vocabulary; the illustration shows a few dimensions and possible outputs.</figcaption>
</figure>

## Turning characters into model inputs

I load the public [makemore names dataset](https://github.com/karpathy/makemore/blob/master/names.txt), map each character to an integer, and reserve index `0` for `.`. A context is therefore a three-integer tensor. The notebook splits whole names into training, development, and test sets before building examples, so examples from one name do not land in different splits.

An embedding matrix gives each character a learned vector. With a 10-dimensional embedding, three characters become 30 numbers. The notebook first uses a tiny two-dimensional matrix to show that indexing a row is equivalent to multiplying by a one-hot vector, then trains a separate 10-dimensional matrix.

```python
embeddings = C[Xtr[batch]]
hidden = torch.tanh(embeddings.reshape(batch_size, -1) @ W1 + b1)
logits = hidden @ W2 + b2
loss = F.cross_entropy(logits, Ytr[batch])
```

The network turns those 30 numbers into scores for every possible next character. Cross-entropy measures how much probability it gave the correct one. Backpropagation then updates the embeddings and weights.

## What I check while training

The notebook plots batch loss and reports loss on the training, development, and test sets. A falling batch loss shows that optimization is working, but the development loss matters for checking whether the model generalizes beyond the names it saw during training. The test set is for a final check.

Once trained, I start from `...`, sample a character from the predicted probabilities, slide the context window forward, and repeat until the model picks `.`. The generated names are a way to inspect what it learned, not proof that the model understands names.

## What I want to try next

I want to compare this network directly with a one-character bigram baseline, tune the learning rate and embedding size, and see how the development loss changes. That comparison should make the value of a longer context easier to measure.
