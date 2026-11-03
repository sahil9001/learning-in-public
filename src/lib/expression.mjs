const functions = Object.freeze({
  sin: Math.sin,
  cos: Math.cos,
  tanh: Math.tanh,
  exp: Math.exp,
  abs: Math.abs,
  sqrt: Math.sqrt,
});

export function compileExpression(source) {
  if (typeof source !== 'string' || !source.trim() || source.length > 120) {
    throw new Error('Enter a function of x (up to 120 characters).');
  }
  const tokens = source.match(/(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?|[a-zA-Z]+|[()+\-*/^]/g) ?? [];
  if (tokens.join('') !== source.replace(/\s+/g, '')) {
    throw new Error('Use numbers, x, arithmetic, parentheses, and supported functions only.');
  }
  let index = 0;
  const peek = () => tokens[index];
  const take = () => tokens[index++];

  function primary() {
    const token = take();
    if (token === '(') {
      const node = expression();
      if (take() !== ')') throw new Error('Close every parenthesis.');
      return node;
    }
    if (token === 'x') return (x) => x;
    if (token === 'pi') return () => Math.PI;
    if (token === 'e') return () => Math.E;
    if (token in functions) {
      if (take() !== '(') throw new Error(`Write ${token}(x) with parentheses.`);
      const argument = expression();
      if (take() !== ')') throw new Error('Close every function call.');
      return (x) => functions[token](argument(x));
    }
    if (token && /^(?:\d|\.)/.test(token)) {
      const number = Number(token);
      return () => number;
    }
    throw new Error('Check the function syntax.');
  }

  function power() {
    const left = primary();
    if (peek() === '^') {
      take();
      const right = unary();
      return (x) => left(x) ** right(x);
    }
    return left;
  }

  function unary() {
    if (peek() === '+') { take(); return unary(); }
    if (peek() === '-') { take(); const node = unary(); return (x) => -node(x); }
    return power();
  }

  function term() {
    let node = unary();
    while (peek() === '*' || peek() === '/') {
      const operator = take();
      const left = node;
      const right = unary();
      node = operator === '*' ? (x) => left(x) * right(x) : (x) => left(x) / right(x);
    }
    return node;
  }

  function expression() {
    let node = term();
    while (peek() === '+' || peek() === '-') {
      const operator = take();
      const left = node;
      const right = term();
      node = operator === '+' ? (x) => left(x) + right(x) : (x) => left(x) - right(x);
    }
    return node;
  }

  const compiled = expression();
  if (index !== tokens.length) throw new Error('Check the function syntax.');
  return (x) => {
    if (!Number.isFinite(x)) throw new Error('x must be a finite number.');
    const result = compiled(x);
    if (!Number.isFinite(result) || Math.abs(result) > 1e6) {
      throw new Error('The function must return a finite value within the selected range.');
    }
    return result;
  };
}
