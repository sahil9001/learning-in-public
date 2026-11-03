import test from 'node:test';
import assert from 'node:assert/strict';
import { compileExpression } from '../src/lib/expression.mjs';
import { Value, MLP, createTrainingSet, trainStep } from '../src/lib/micrograd.mjs';

test('expression parser supports the notebook quadratic and functions', () => {
  assert.equal(compileExpression('3*x^2 - 4*x + 5')(3), 20);
  assert.ok(Math.abs(compileExpression('sin(x) + x/2')(Math.PI) - Math.PI / 2) < 1e-10);
  assert.equal(compileExpression('-(x+2)^2')(-1), -1);
});

test('expression parser rejects code and invalid input', () => {
  for (const input of ['x.constructor', 'alert(1)', 'x;1', '2 ** 3', '', 'sin(', 'x/0']) {
    assert.throws(() => compileExpression(input)(1), input);
  }
});

test('autograd accumulates gradients through shared nodes', () => {
  const x = new Value(3);
  const y = x.mul(x).add(x);
  y.backward();
  assert.equal(y.data, 12);
  assert.equal(x.grad, 7);
});

test('autograd applies the tanh and power chain rules', () => {
  const x = new Value(0.5);
  const output = x.pow(2).tanh();
  output.backward();
  assert.ok(Math.abs(x.grad - (1 - Math.tanh(0.25) ** 2)) < 1e-10);
});

test('training samples are finite, normalized, and range checked', () => {
  const samples = createTrainingSet(compileExpression('3*x^2 - 4*x + 5'), -2, 2, 9);
  assert.equal(samples.length, 9);
  assert.equal(samples[0].target, 25);
  assert.ok(samples.every(({ y }) => Math.abs(y) <= 1));
  assert.throws(() => createTrainingSet(() => 1, 2, -2));
  assert.throws(() => createTrainingSet(compileExpression('1/x'), -1, 1, 9));
});

test('training a small MLP reduces mean squared error', () => {
  let seed = 42;
  const random = () => { seed = (1664525 * seed + 1013904223) % 4294967296; return seed / 4294967296; };
  const model = new MLP(1, [6, 1], random);
  const samples = createTrainingSet(compileExpression('x^2'), -1, 1, 15);
  const first = trainStep(model, samples, 0.05);
  let last = first;
  for (let i = 0; i < 300; i++) last = trainStep(model, samples, 0.05);
  assert.ok(last < first * 0.6, `loss stayed at ${last} from ${first}`);
});
