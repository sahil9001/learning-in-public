export class Value {
  constructor(data, parents = [], backward = () => {}) {
    this.data = data;
    this.grad = 0;
    this.parents = parents;
    this.localBackward = backward;
  }

  add(other) {
    const b = other instanceof Value ? other : new Value(other);
    const out = new Value(this.data + b.data, [this, b]);
    out.localBackward = () => { this.grad += out.grad; b.grad += out.grad; };
    return out;
  }

  mul(other) {
    const b = other instanceof Value ? other : new Value(other);
    const out = new Value(this.data * b.data, [this, b]);
    out.localBackward = () => {
      this.grad += b.data * out.grad;
      b.grad += this.data * out.grad;
    };
    return out;
  }

  pow(exponent) {
    const out = new Value(this.data ** exponent, [this]);
    out.localBackward = () => { this.grad += exponent * this.data ** (exponent - 1) * out.grad; };
    return out;
  }

  tanh() {
    const t = Math.tanh(this.data);
    const out = new Value(t, [this]);
    out.localBackward = () => { this.grad += (1 - t * t) * out.grad; };
    return out;
  }

  backward() {
    const order = [];
    const seen = new Set();
    const visit = (node) => {
      if (seen.has(node)) return;
      seen.add(node);
      node.parents.forEach(visit);
      order.push(node);
    };
    visit(this);
    this.grad = 1;
    for (let i = order.length - 1; i >= 0; i--) order[i].localBackward();
  }
}

export class MLP {
  constructor(inputSize, layerSizes, random = Math.random) {
    const sizes = [inputSize, ...layerSizes];
    this.layers = layerSizes.map((size, layerIndex) =>
      Array.from({ length: size }, () => ({
        weights: Array.from({ length: sizes[layerIndex] }, () => new Value((random() * 2 - 1) / Math.sqrt(sizes[layerIndex]))),
        bias: new Value((random() * 2 - 1) * 0.1),
      }))
    );
  }

  parameters() {
    return this.layers.flatMap((layer) => layer.flatMap((neuron) => [...neuron.weights, neuron.bias]));
  }

  predict(x, trace = false) {
    let activations = [new Value(x)];
    const layers = [activations.map((value) => value.data)];
    for (let i = 0; i < this.layers.length; i++) {
      activations = this.layers[i].map((neuron) => {
        const sum = neuron.weights.reduce((total, weight, j) => total.add(weight.mul(activations[j])), neuron.bias);
        return i === this.layers.length - 1 ? sum : sum.tanh();
      });
      layers.push(activations.map((value) => value.data));
    }
    return trace ? { output: activations[0].data, layers } : activations[0];
  }
}

export function createTrainingSet(fn, min, max, count = 25) {
  if (!Number.isFinite(min) || !Number.isFinite(max) || min >= max || count < 2 || count > 100) {
    throw new Error('Choose a valid x range and sample count.');
  }
  const raw = Array.from({ length: count }, (_, index) => {
    const x = min + (max - min) * index / (count - 1);
    return { x, target: fn(x) };
  });
  const values = raw.map((sample) => sample.target);
  const center = (Math.min(...values) + Math.max(...values)) / 2;
  const scale = Math.max(1, (Math.max(...values) - Math.min(...values)) / 2);
  return raw.map((sample) => ({ ...sample, y: (sample.target - center) / scale, center, scale }));
}

export function trainStep(model, samples, rate) {
  if (!Number.isFinite(rate) || rate <= 0 || rate > 0.5) throw new Error('Learning rate must be between 0 and 0.5.');
  const losses = samples.map(({ x, y }) => model.predict(x).add(-y).pow(2));
  const loss = losses.reduce((sum, item) => sum.add(item), new Value(0)).mul(1 / samples.length);
  for (const parameter of model.parameters()) parameter.grad = 0;
  loss.backward();
  if (!Number.isFinite(loss.data)) throw new Error('Training became unstable. Try a lower learning rate.');
  for (const parameter of model.parameters()) {
    const gradient = Math.max(-10, Math.min(10, parameter.grad));
    parameter.data -= rate * gradient;
    if (!Number.isFinite(parameter.data)) throw new Error('Training became unstable. Try a lower learning rate.');
  }
  return loss.data;
}
