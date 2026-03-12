/**
 * NeuralNetwork — Pure JavaScript fully-connected neural network.
 * Supports forward pass, backpropagation, and gradient descent.
 * Designed for 2D binary classification.
 */
class NeuralNetwork {
  /**
   * @param {number[]} layerSizes  e.g. [2, 4, 4, 1]
   * @param {string}   activation  'relu' | 'sigmoid' | 'tanh' | 'linear'
   */
  constructor(layerSizes, activation = 'relu') {
    this.layerSizes = [...layerSizes];
    this.activation = activation;
    this.weights    = [];   // weights[l][j][k] = w from neuron k (layer l) to j (layer l+1)
    this.biases     = [];   // biases[l][j]     = bias of neuron j in layer l+1
    this._init();
  }

  /** He-initialized weights, zero biases */
  _init() {
    this.weights = [];
    this.biases  = [];
    for (let l = 0; l < this.layerSizes.length - 1; l++) {
      const fi = this.layerSizes[l];
      const fo = this.layerSizes[l + 1];
      const sc = Math.sqrt(2.0 / fi);
      this.weights.push(
        Array.from({ length: fo }, () =>
          Array.from({ length: fi }, () => (Math.random() * 2 - 1) * sc)
        )
      );
      this.biases.push(new Array(fo).fill(0));
    }
  }

  // ── Activation functions ───────────────────────────────────────────────────

  _clip(z) { return Math.max(-500, Math.min(500, z)); }

  _act(z) {
    switch (this.activation) {
      case 'relu':    return z > 0 ? z : 0;
      case 'sigmoid': return 1 / (1 + Math.exp(-this._clip(z)));
      case 'tanh':    return Math.tanh(z);
      default:        return z;
    }
  }

  _actD(z) {   // derivative of hidden activation
    switch (this.activation) {
      case 'relu':    return z > 0 ? 1 : 0;
      case 'sigmoid': { const s = 1 / (1 + Math.exp(-this._clip(z))); return s * (1 - s); }
      case 'tanh':    { const t = Math.tanh(z); return 1 - t * t; }
      default:        return 1;
    }
  }

  _sig(z) { return 1 / (1 + Math.exp(-this._clip(z))); } // output-layer sigmoid

  // ── Forward pass ──────────────────────────────────────────────────────────

  /**
   * Run a forward pass.
   * @param   {number[]} input   [x, y]
   * @returns {{ A: number[][], Z: number[][], out: number }}
   *          A[l] = activations at layer l (A[0] = input)
   *          Z[l] = pre-activations produced by weight layer l
   */
  forward(input) {
    const A = [input.slice()];
    const Z = [];

    for (let l = 0; l < this.weights.length; l++) {
      const isOut = (l === this.weights.length - 1);
      const prevA = A[l];
      const z = [], a = [];

      for (let j = 0; j < this.layerSizes[l + 1]; j++) {
        let s = this.biases[l][j];
        for (let k = 0; k < this.layerSizes[l]; k++) {
          s += this.weights[l][j][k] * prevA[k];
        }
        z.push(s);
        a.push(isOut ? this._sig(s) : this._act(s));
      }
      Z.push(z);
      A.push(a);
    }

    return { A, Z, out: A[A.length - 1][0] };
  }

  predict(x, y) { return this.forward([x, y]).out; }

  // ── Loss / Accuracy ────────────────────────────────────────────────────────

  loss(pts) {
    if (!pts.length) return 0;
    let L = 0;
    for (const p of pts) {
      const o = this.predict(p.x, p.y);
      L -= p.label * Math.log(o + 1e-10) + (1 - p.label) * Math.log(1 - o + 1e-10);
    }
    return L / pts.length;
  }

  accuracy(pts) {
    if (!pts.length) return 0;
    return pts.filter(p =>
      (this.predict(p.x, p.y) > 0.5 ? 1 : 0) === p.label
    ).length / pts.length;
  }

  // ── Backpropagation & weight update ───────────────────────────────────────

  /**
   * One gradient descent step over all data points.
   * @param   {Array}  pts         data points { x, y, label }
   * @param   {number} lr          learning rate
   * @returns {number}             mean cross-entropy loss
   */
  step(pts, lr) {
    if (!pts.length) return 0;

    const n  = pts.length;
    const nL = this.weights.length;

    // Gradient accumulators
    const gW = this.weights.map(W => W.map(row => row.map(() => 0)));
    const gB = this.biases.map(b => b.map(() => 0));
    let totLoss = 0;

    for (const p of pts) {
      const { A, Z } = this.forward([p.x, p.y]);
      const out = A[A.length - 1][0];

      totLoss -= p.label * Math.log(out + 1e-10) + (1 - p.label) * Math.log(1 - out + 1e-10);

      // Deltas — δ[l] corresponds to weight-layer l
      const delta = new Array(nL);
      delta[nL - 1] = [out - p.label];   // d(BCE) / d(z_out) = output − label

      for (let l = nL - 2; l >= 0; l--) {
        const dl = [];
        for (let k = 0; k < this.layerSizes[l + 1]; k++) {
          let g = 0;
          for (let j = 0; j < this.layerSizes[l + 2]; j++) {
            g += this.weights[l + 1][j][k] * delta[l + 1][j];
          }
          dl.push(g * this._actD(Z[l][k]));
        }
        delta[l] = dl;
      }

      // Accumulate gradients
      for (let l = 0; l < nL; l++) {
        for (let j = 0; j < this.layerSizes[l + 1]; j++) {
          gB[l][j] += delta[l][j];
          for (let k = 0; k < this.layerSizes[l]; k++) {
            gW[l][j][k] += delta[l][j] * A[l][k];
          }
        }
      }
    }

    // Apply updates with gradient clipping
    const clip = 5.0;
    for (let l = 0; l < nL; l++) {
      for (let j = 0; j < this.layerSizes[l + 1]; j++) {
        this.biases[l][j] -= lr * Math.max(-clip, Math.min(clip, gB[l][j] / n));
        for (let k = 0; k < this.layerSizes[l]; k++) {
          this.weights[l][j][k] -= lr * Math.max(-clip, Math.min(clip, gW[l][j][k] / n));
        }
      }
    }

    return totLoss / n;
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  get totalParams() {
    let t = 0;
    for (let l = 0; l < this.weights.length; l++) {
      t += this.layerSizes[l + 1] * this.layerSizes[l] + this.layerSizes[l + 1];
    }
    return t;
  }

  get layerCount() { return this.layerSizes.length; }
}
