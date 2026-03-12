/**
 * app.js — Neural Network Visualizer
 * Handles UI, canvas rendering, and training loop.
 */

(function () {
  'use strict';

  // ══════════════════════════════════════════════════════════════════════════
  //  Constants & Palette
  // ══════════════════════════════════════════════════════════════════════════

  const CANVAS_W = 400;
  const CANVAS_H = 400;
  const NET_W    = 400;
  const NET_H    = 180;
  const LOSS_W   = 400;
  const LOSS_H   = 90;
  const RANGE    = 1.2;       // data space: [-RANGE, RANGE] on each axis
  const BD_RES   = 110;       // decision-boundary grid resolution
  const MAX_HIST = 400;       // max loss-history points

  const C = {
    bg:       [7,   7,  18],
    classA:   [0,   255, 153],   // #00ff99
    classB:   [255, 51,  102],   // #ff3366
    boundLine:[255, 255, 255],
    text:     '#c8d4e8',
    dim:      '#4a5580',
    accent:   '#00e5ff',
    posConn:  'rgba(0,229,255,',
    negConn:  'rgba(255,100,60,',
    neutrConn:'rgba(80,80,120,',
  };

  // ══════════════════════════════════════════════════════════════════════════
  //  State
  // ══════════════════════════════════════════════════════════════════════════

  let nn       = null;
  let pts      = [];       // { x, y, label (0|1) }
  let training = false;
  let rafId    = null;
  let epoch    = 0;
  let lossHist = [];
  let lastFwd  = null;     // last forward-pass result (for diagram colouring)
  let bdDirty  = true;     // flag: recompute boundary on next frame
  let bdPixels = null;     // Float32Array of predictions on the grid

  let cfg = {
    hiddenLayers:    2,
    neuronsPerLayer: 4,
    activation:      'relu',
    lr:              0.1,
    stepsPerFrame:   5,
  };

  // ══════════════════════════════════════════════════════════════════════════
  //  Canvas contexts
  // ══════════════════════════════════════════════════════════════════════════

  let mCtx, nCtx, lCtx;

  // ══════════════════════════════════════════════════════════════════════════
  //  Boot
  // ══════════════════════════════════════════════════════════════════════════

  window.addEventListener('DOMContentLoaded', () => {
    const mCanvas = document.getElementById('mainCanvas');
    const nCanvas = document.getElementById('networkCanvas');
    const lCanvas = document.getElementById('lossCanvas');

    mCanvas.width  = CANVAS_W; mCanvas.height = CANVAS_H;
    nCanvas.width  = NET_W;    nCanvas.height = NET_H;
    lCanvas.width  = LOSS_W;   lCanvas.height = LOSS_H;

    mCtx = mCanvas.getContext('2d');
    nCtx = nCanvas.getContext('2d');
    lCtx = lCanvas.getContext('2d');

    bindUI(mCanvas);
    buildNetwork();

    // Continuous render loop — always runs, does training steps when active
    (function loop() {
      if (training) doSteps(cfg.stepsPerFrame);
      renderAll();
      requestAnimationFrame(loop);
    })();
  });

  // ══════════════════════════════════════════════════════════════════════════
  //  Network construction
  // ══════════════════════════════════════════════════════════════════════════

  function buildNetwork() {
    const sizes = [2];
    for (let i = 0; i < cfg.hiddenLayers; i++) sizes.push(cfg.neuronsPerLayer);
    sizes.push(1);
    nn       = new NeuralNetwork(sizes, cfg.activation);
    epoch    = 0;
    lossHist = [];
    bdDirty  = true;
    generateSliders();
    updateArchLabel(sizes);
    updateStats(null);
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  UI Bindings
  // ══════════════════════════════════════════════════════════════════════════

  function bindUI(canvas) {
    // ── Canvas click events ──────────────────────────────────────────────
    canvas.addEventListener('click', e => { e.preventDefault(); placePoint(e, 1); });
    canvas.addEventListener('contextmenu', e => { e.preventDefault(); placePoint(e, 0); });
    canvas.addEventListener('touchstart', e => {
      e.preventDefault();
      placePoint(e.touches[0], 1);
    }, { passive: false });

    // ── Architecture ─────────────────────────────────────────────────────
    on('numLayers', 'input', function () {
      cfg.hiddenLayers = +this.value;
      t('numLayersVal', this.value);
    });
    on('numNeurons', 'input', function () {
      cfg.neuronsPerLayer = +this.value;
      t('numNeuronsVal', this.value);
    });
    on('activation', 'change', function () {
      cfg.activation = this.value;
      if (nn) { nn.activation = this.value; bdDirty = true; }
    });
    on('rebuildBtn', 'click', () => {
      training = false;
      updateTrainBtn();
      buildNetwork();
    });

    // ── Training ─────────────────────────────────────────────────────────
    on('lrSlider', 'input', function () {
      cfg.lr = parseFloat(Math.pow(10, +this.value).toPrecision(4));
      t('lrVal', cfg.lr.toFixed(4));
    });
    on('stepsSlider', 'input', function () {
      cfg.stepsPerFrame = +this.value;
      t('stepsVal', this.value);
    });
    on('trainBtn', 'click', () => {
      if (!pts.length) { flashNoData(); return; }
      training = !training;
      updateTrainBtn();
    });
    on('stepBtn', 'click', () => {
      if (!pts.length || !nn) { flashNoData(); return; }
      doSteps(1);
    });
    on('resetWeightsBtn', 'click', () => {
      if (!nn) return;
      nn._init();
      epoch = 0; lossHist = []; bdDirty = true;
      syncSlidersFromNN();
      updateStats(null);
    });
    on('clearDataBtn', 'click', () => {
      pts     = [];
      training = false;
      updateTrainBtn();
      bdDirty = true;
      updateStats(null);
    });
    on('randomizeWeightsBtn', 'click', () => {
      if (!nn) return;
      nn._init();
      syncSlidersFromNN();
      bdDirty = true;
    });
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  Data point helpers
  // ══════════════════════════════════════════════════════════════════════════

  function placePoint(e, label) {
    const canvas = document.getElementById('mainCanvas');
    const rect   = canvas.getBoundingClientRect();
    const px     = e.clientX - rect.left;
    const py     = e.clientY - rect.top;
    const x      =  (px / rect.width)  * 2 * RANGE - RANGE;
    const y      = -(py / rect.height) * 2 * RANGE + RANGE;  // flip Y
    pts.push({ x, y, label });
    bdDirty = true;
    updateStats(null);
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  Training loop
  // ══════════════════════════════════════════════════════════════════════════

  function doSteps(n) {
    for (let i = 0; i < n; i++) {
      const loss = nn.step(pts, cfg.lr);
      epoch++;
      if (i === 0) {
        lossHist.push(loss);
        if (lossHist.length > MAX_HIST) lossHist.shift();
        updateStats(loss);
      }
    }
    bdDirty = true;
    syncSlidersFromNN();
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  Decision boundary — compute on grid
  // ══════════════════════════════════════════════════════════════════════════

  function computeBoundary() {
    if (!nn) return;
    bdPixels = new Float32Array(BD_RES * BD_RES);
    for (let row = 0; row < BD_RES; row++) {
      for (let col = 0; col < BD_RES; col++) {
        const x = (col / (BD_RES - 1)) * 2 * RANGE - RANGE;
        const y = RANGE - (row / (BD_RES - 1)) * 2 * RANGE;
        bdPixels[row * BD_RES + col] = nn.predict(x, y);
      }
    }
    bdDirty = false;
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  Render — main canvas
  // ══════════════════════════════════════════════════════════════════════════

  function renderAll() {
    if (bdDirty) computeBoundary();
    renderMain();
    renderNetwork();
    renderLoss();
  }

  function renderMain() {
    const ctx = mCtx;
    const W = CANVAS_W, H = CANVAS_H;

    // Background
    ctx.fillStyle = `rgb(${C.bg.join(',')})`;
    ctx.fillRect(0, 0, W, H);

    // Decision boundary as ImageData
    if (bdPixels) {
      const imgData = ctx.createImageData(BD_RES, BD_RES);
      const d = imgData.data;
      for (let i = 0; i < BD_RES * BD_RES; i++) {
        const p    = bdPixels[i];            // probability ∈ [0,1]
        const conf = Math.abs(p - 0.5) * 2; // 0=unsure, 1=certain
        let r, g, b;
        if (p >= 0.5) {
          // Class A — cyan-green
          r = lerpInt(C.bg[0], C.classA[0], conf * 0.72);
          g = lerpInt(C.bg[1], C.classA[1], conf * 0.72);
          b = lerpInt(C.bg[2], C.classA[2], conf * 0.72);
        } else {
          // Class B — pink-red
          r = lerpInt(C.bg[0], C.classB[0], conf * 0.72);
          g = lerpInt(C.bg[1], C.classB[1], conf * 0.72);
          b = lerpInt(C.bg[2], C.classB[2], conf * 0.72);
        }
        d[i * 4]     = r;
        d[i * 4 + 1] = g;
        d[i * 4 + 2] = b;
        d[i * 4 + 3] = 255;
      }
      // Draw scaled up to full canvas
      const offCanvas = new OffscreenCanvas(BD_RES, BD_RES);
      const offCtx    = offCanvas.getContext('2d');
      offCtx.putImageData(imgData, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'medium';
      ctx.drawImage(offCanvas, 0, 0, W, H);
    }

    // Grid lines (subtle)
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth   = 1;
    for (let i = 1; i < 4; i++) {
      ctx.beginPath(); ctx.moveTo(W * i / 4, 0); ctx.lineTo(W * i / 4, H); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, H * i / 4); ctx.lineTo(W, H * i / 4); ctx.stroke();
    }
    // Axes
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth   = 1;
    ctx.beginPath(); ctx.moveTo(W / 2, 0); ctx.lineTo(W / 2, H); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke();

    // Data points
    const ptR = 6;
    for (const p of pts) {
      const px = dataToCanvasX(p.x);
      const py = dataToCanvasY(p.y);
      // Glow
      const col = p.label === 1
        ? `rgba(${C.classA.join(',')},`
        : `rgba(${C.classB.join(',')},`;
      const grd = ctx.createRadialGradient(px, py, 0, px, py, ptR * 2.8);
      grd.addColorStop(0, col + '0.4)');
      grd.addColorStop(1, col + '0)');
      ctx.beginPath(); ctx.arc(px, py, ptR * 2.8, 0, Math.PI * 2);
      ctx.fillStyle = grd; ctx.fill();
      // Point
      ctx.beginPath(); ctx.arc(px, py, ptR, 0, Math.PI * 2);
      ctx.fillStyle   = p.label === 1 ? `rgb(${C.classA.join(',')})` : `rgb(${C.classB.join(',')})`;
      ctx.shadowColor = p.label === 1 ? `rgb(${C.classA.join(',')})` : `rgb(${C.classB.join(',')})`;
      ctx.shadowBlur  = 8;
      ctx.fill();
      ctx.shadowBlur  = 0;
      // Border
      ctx.strokeStyle = 'rgba(255,255,255,0.5)';
      ctx.lineWidth   = 1;
      ctx.stroke();
    }

    // Empty-state message
    if (!pts.length) {
      ctx.fillStyle = 'rgba(255,255,255,0.18)';
      ctx.font      = '13px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText('Left-click → Class A   |   Right-click → Class B', W / 2, H / 2 + 2);
      ctx.textAlign = 'left';
    }
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  Render — network diagram
  // ══════════════════════════════════════════════════════════════════════════

  function renderNetwork() {
    if (!nn) return;
    const ctx = nCtx;
    const W = NET_W, H = NET_H;

    ctx.fillStyle = `rgb(${C.bg.join(',')})`;
    ctx.fillRect(0, 0, W, H);

    const sizes  = nn.layerSizes;
    const nL     = sizes.length;
    const padX   = 30;
    const layerX = i => padX + (i / (nL - 1)) * (W - padX * 2);

    // Get activations for a sample point (last data point or origin)
    let sampleA = null;
    if (pts.length > 0) {
      const sp  = pts[pts.length - 1];
      sampleA   = nn.forward([sp.x, sp.y]).A;
    } else {
      sampleA = nn.forward([0, 0]).A;
    }

    // Neuron positions
    const pos = sizes.map((sz, l) => {
      const cx = layerX(l);
      return Array.from({ length: sz }, (_, j) => {
        const padY = 20;
        const span = H - padY * 2;
        const y    = sz === 1 ? H / 2 : padY + (j / (sz - 1)) * span;
        return { cx, cy: y };
      });
    });

    const maxR = 13;
    const neuronR = sizes.some(s => s > 6) ? 9 : maxR;

    // ── Connections ──────────────────────────────────────────────────────
    for (let l = 0; l < nL - 1; l++) {
      for (let j = 0; j < sizes[l + 1]; j++) {
        for (let k = 0; k < sizes[l]; k++) {
          const w    = nn.weights[l][j][k];
          const absW = Math.min(Math.abs(w), 4) / 4;
          const alpha = 0.08 + absW * 0.55;
          const colStr = w > 0 ? C.posConn : (w < 0 ? C.negConn : C.neutrConn);
          ctx.strokeStyle = colStr + alpha + ')';
          ctx.lineWidth   = 0.5 + absW * 2.5;
          ctx.beginPath();
          ctx.moveTo(pos[l][k].cx, pos[l][k].cy);
          ctx.lineTo(pos[l + 1][j].cx, pos[l + 1][j].cy);
          ctx.stroke();
        }
      }
    }

    // ── Neurons ──────────────────────────────────────────────────────────
    for (let l = 0; l < nL; l++) {
      for (let j = 0; j < sizes[l]; j++) {
        const { cx, cy } = pos[l][j];
        const act = sampleA ? sampleA[l][j] : 0;

        let fillColor;
        if (l === 0) {
          fillColor = `rgba(180,200,255,0.9)`;  // input — white-blue
        } else if (l === nL - 1) {
          // Output — interpolate A↔B
          fillColor = `rgb(${
            lerpInt(C.classB[0], C.classA[0], act)},${
            lerpInt(C.classB[1], C.classA[1], act)},${
            lerpInt(C.classB[2], C.classA[2], act)})`;
        } else {
          // Hidden — activation-based brightness in cyan
          const t = Math.max(0, Math.min(1, act));
          fillColor = `rgba(${lerpInt(20, C.classA[0], t)},${lerpInt(30, C.classA[1], t)},${lerpInt(60, C.classA[2], t)},0.95)`;
        }

        // Shadow
        ctx.shadowColor = fillColor;
        ctx.shadowBlur  = act > 0.3 ? 8 : 2;

        // Circle
        ctx.beginPath();
        ctx.arc(cx, cy, neuronR, 0, Math.PI * 2);
        ctx.fillStyle   = fillColor;
        ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.3)';
        ctx.lineWidth   = 1;
        ctx.stroke();
        ctx.shadowBlur  = 0;

        // Label (only for small networks)
        if (sizes[l] <= 6 && nL <= 5) {
          ctx.fillStyle = 'rgba(255,255,255,0.55)';
          ctx.font      = '8px "JetBrains Mono", monospace';
          ctx.textAlign = 'center';
          ctx.fillText(act.toFixed(2), cx, cy + neuronR + 10);
        }
      }
    }

    // ── Layer labels ─────────────────────────────────────────────────────
    for (let l = 0; l < nL; l++) {
      ctx.fillStyle = C.dim;
      ctx.font      = '9px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      const label   = l === 0 ? 'INPUT' : (l === nL - 1 ? 'OUTPUT' : `HIDDEN ${l}`);
      ctx.fillText(label, layerX(l), H - 4);
    }
    ctx.textAlign = 'left';
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  Render — loss chart
  // ══════════════════════════════════════════════════════════════════════════

  function renderLoss() {
    const ctx = lCtx;
    const W = LOSS_W, H = LOSS_H;
    const pad = { t: 8, r: 10, b: 22, l: 40 };

    ctx.fillStyle = `rgb(${C.bg.join(',')})`;
    ctx.fillRect(0, 0, W, H);

    if (lossHist.length < 2) {
      ctx.fillStyle = C.dim;
      ctx.font      = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText('Loss will appear here during training', W / 2, H / 2 + 4);
      ctx.textAlign = 'left';
      return;
    }

    const maxL = Math.max(...lossHist, 0.001);
    const minL = 0;
    const cW   = W - pad.l - pad.r;
    const cH   = H - pad.t - pad.b;

    // Grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth   = 1;
    for (let i = 1; i <= 3; i++) {
      const y = pad.t + cH * (i / 4);
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    }

    // Filled area
    ctx.beginPath();
    ctx.moveTo(pad.l, pad.t + cH);
    for (let i = 0; i < lossHist.length; i++) {
      const x = pad.l + (i / (MAX_HIST - 1)) * cW;
      const y = pad.t + cH - ((lossHist[i] - minL) / (maxL - minL)) * cH;
      ctx.lineTo(x, y);
    }
    ctx.lineTo(pad.l + ((lossHist.length - 1) / (MAX_HIST - 1)) * cW, pad.t + cH);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, pad.t, 0, pad.t + cH);
    grad.addColorStop(0, 'rgba(0,229,255,0.35)');
    grad.addColorStop(1, 'rgba(0,229,255,0.02)');
    ctx.fillStyle = grad;
    ctx.fill();

    // Line
    ctx.beginPath();
    for (let i = 0; i < lossHist.length; i++) {
      const x = pad.l + (i / (MAX_HIST - 1)) * cW;
      const y = pad.t + cH - ((lossHist[i] - minL) / (maxL - minL)) * cH;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.strokeStyle = C.accent;
    ctx.lineWidth   = 1.5;
    ctx.stroke();

    // Y axis labels
    ctx.fillStyle = C.dim;
    ctx.font      = '9px "JetBrains Mono", monospace';
    ctx.textAlign = 'right';
    ctx.fillText(maxL.toFixed(2), pad.l - 4, pad.t + 5);
    ctx.fillText('0.00', pad.l - 4, pad.t + cH + 4);
    ctx.textAlign = 'left';

    // X label
    ctx.fillStyle = C.dim;
    ctx.font      = '9px "JetBrains Mono", monospace';
    ctx.fillText('epoch', pad.l, H - 4);
    ctx.textAlign = 'right';
    ctx.fillText(epoch.toString(), W - pad.r, H - 4);
    ctx.textAlign = 'left';
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  Weight / Bias sliders
  // ══════════════════════════════════════════════════════════════════════════

  const SLIDER_MAX   = 4;
  const SLIDER_MIN   = -4;
  const SLIDER_STEPS = 400;

  function generateSliders() {
    const container = document.getElementById('slidersContainer');
    container.innerHTML = '';
    if (!nn) return;

    const total = nn.totalParams;
    if (total > 80) {
      container.innerHTML = `
        <div class="slider-overflow-msg">
          Network has <strong>${total}</strong> parameters — too many to show individual sliders.
          Use training to optimise weights.
        </div>`;
      return;
    }

    for (let l = 0; l < nn.weights.length; l++) {
      const fi = nn.layerSizes[l];
      const fo = nn.layerSizes[l + 1];

      const sec = document.createElement('div');
      sec.className = 'slider-layer-section';

      const hdr = document.createElement('div');
      hdr.className = 'slider-layer-hdr';
      hdr.textContent = `Layer ${l} → ${l + 1}  (${fi}→${fo})`;
      sec.appendChild(hdr);

      // Weights
      for (let j = 0; j < fo; j++) {
        for (let k = 0; k < fi; k++) {
          sec.appendChild(makeSlider(
            `w_${l}_${j}_${k}`,
            `w[${j}←${k}]`,
            nn.weights[l][j][k],
            v => { nn.weights[l][j][k] = v; bdDirty = true; }
          ));
        }
      }

      // Biases
      for (let j = 0; j < fo; j++) {
        sec.appendChild(makeSlider(
          `b_${l}_${j}`,
          `b[${j}]`,
          nn.biases[l][j],
          v => { nn.biases[l][j] = v; bdDirty = true; }
        ));
      }

      container.appendChild(sec);
    }
  }

  function makeSlider(id, label, initVal, onChange) {
    const row = document.createElement('div');
    row.className = 'weight-slider-row';

    const lbl = document.createElement('span');
    lbl.className = 'weight-slider-label';
    lbl.textContent = label;

    const input = document.createElement('input');
    input.type    = 'range';
    input.id      = id;
    input.min     = SLIDER_MIN;
    input.max     = SLIDER_MAX;
    input.step    = (SLIDER_MAX - SLIDER_MIN) / SLIDER_STEPS;
    input.value   = initVal;
    input.className = 'weight-slider';

    const valSpan = document.createElement('span');
    valSpan.className = 'weight-slider-val';
    valSpan.id        = id + '_val';
    valSpan.textContent = fmtW(initVal);

    input.addEventListener('input', function () {
      const v = parseFloat(this.value);
      valSpan.textContent = fmtW(v);
      onChange(v);
      // Re-render immediately (no training tick needed)
      renderAll();
    });

    row.appendChild(lbl);
    row.appendChild(input);
    row.appendChild(valSpan);
    return row;
  }

  function syncSlidersFromNN() {
    if (!nn) return;
    for (let l = 0; l < nn.weights.length; l++) {
      for (let j = 0; j < nn.layerSizes[l + 1]; j++) {
        for (let k = 0; k < nn.layerSizes[l]; k++) {
          setSlider(`w_${l}_${j}_${k}`, nn.weights[l][j][k]);
        }
        setSlider(`b_${l}_${j}`, nn.biases[l][j]);
      }
    }
  }

  function setSlider(id, val) {
    const el = document.getElementById(id);
    const vEl = document.getElementById(id + '_val');
    if (el)  el.value = val;
    if (vEl) vEl.textContent = fmtW(val);
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  UI helpers
  // ══════════════════════════════════════════════════════════════════════════

  function updateTrainBtn() {
    const btn = document.getElementById('trainBtn');
    btn.textContent = training ? '⏸ Pause' : '▶ Train';
    btn.classList.toggle('btn-active', training);

    const pill = document.getElementById('statusPill');
    if (pill) {
      pill.textContent    = training ? 'TRAINING' : 'IDLE';
      pill.dataset.state  = training ? 'training' : 'idle';
    }
  }

  function updateStats(loss) {
    t('epochCount', epoch);

    const classA = pts.filter(p => p.label === 1).length;
    const classB = pts.filter(p => p.label === 0).length;
    t('pointCount', pts.length);
    t('classACount', classA);
    t('classBCount', classB);

    if (loss !== null && pts.length) {
      t('lossDisplay', loss.toFixed(4));
      const acc = nn ? nn.accuracy(pts) : 0;
      t('accuracy', (acc * 100).toFixed(1) + '%');
    } else {
      t('lossDisplay', '—');
      t('accuracy', '—');
    }
  }

  function updateArchLabel(sizes) {
    const el = document.getElementById('archSummary');
    if (el && sizes) el.textContent = sizes.join(' → ');
  }

  function flashNoData() {
    const btn = document.getElementById('trainBtn');
    btn.textContent = '⚠ Add data first';
    setTimeout(() => { btn.textContent = training ? '⏸ Pause' : '▶ Train'; }, 1500);
  }

  function recordLoss(loss) {
    lossHist.push(loss);
    if (lossHist.length > MAX_HIST) lossHist.shift();
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  Coordinate helpers
  // ══════════════════════════════════════════════════════════════════════════

  function dataToCanvasX(x) { return (x + RANGE) / (2 * RANGE) * CANVAS_W; }
  function dataToCanvasY(y) { return (RANGE - y) / (2 * RANGE) * CANVAS_H; }

  // ══════════════════════════════════════════════════════════════════════════
  //  Misc utils
  // ══════════════════════════════════════════════════════════════════════════

  function lerpInt(a, b, t) { return Math.round(a + (b - a) * Math.min(1, Math.max(0, t))); }
  function fmtW(v)           { return (v >= 0 ? ' ' : '') + v.toFixed(3); }
  function on(id, ev, fn)    { const el = document.getElementById(id); if (el) el.addEventListener(ev, fn); }
  function t(id, val)        { const el = document.getElementById(id); if (el) el.textContent = val; }

})();
