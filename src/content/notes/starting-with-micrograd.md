---
title: "Starting with micrograd"
summary: "A first look at derivatives, computation graphs, and why a tiny autograd engine helps make backpropagation less mysterious."
date: 2026-11-04
topic: "Neural networks"
status: "In progress"
readingTime: "3 min read"
---

I'm learning neural networks from the ground up through [Andrej Karpathy's micrograd lecture](https://www.youtube.com/watch?v=VMj-3S1tku0). This notebook is my starting point, not a polished claim that I've mastered the subject.

## The starting question

How can a network tell which weights to change when its prediction is wrong? The lecture approaches that question with ordinary scalar arithmetic, then builds a small engine that remembers how a result was calculated.

In [my Colab notebook](https://colab.research.google.com/drive/1jlF4JoM39M4RcGP04sYwcRpVe1Qud9tA), I start with a quadratic:

```python
def f(x):
    return 3 * x ** 2 - 4 * x + 5
```

Plotting it gives me a visible target. A small change in `x` gives an approximate derivative, which is a useful bridge from the calculus definition to code.

## The path through the notebook

1. **Numerical derivatives:** nudge an input by a small amount and observe how the output moves.
2. **A `Value` object:** store a number, its parents in an expression, and its gradient.
3. **Backpropagation:** walk the computation graph backward, applying local derivatives and accumulating contributions when a node is reused.
4. **Neurons and layers:** combine weighted inputs with a bias, then stack those units into an MLP.
5. **Training:** compare predictions with targets using squared error and take gradient descent steps.

The [interactive lab](../../lab/) turns the last step into a live experiment. Try the quadratic, change the hidden layer size or learning rate, and watch the approximation improve or struggle.

## What I want to investigate next

I want to understand how the learning rate affects convergence, why gradients must be reset between steps, and when a small network fails to fit a curve. I'll keep updating this journal as those answers become clearer.
