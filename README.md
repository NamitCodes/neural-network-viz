# NeuralLab ⬡ — Interactive Neural Network Visualizer

A pure-frontend interactive tool for visualising how neural networks learn and make predictions in real time. No build step required.

![NeuralLab Screenshot](https://img.shields.io/badge/frontend-vanilla%20JS-00e5ff?style=flat-square) ![No dependencies](https://img.shields.io/badge/dependencies-zero-00ff99?style=flat-square) ![License](https://img.shields.io/badge/license-MIT-7b61ff?style=flat-square)

---

## Features

| Feature                       | Details                                                     |
| ----------------------------- | ----------------------------------------------------------- |
| **Interactive data**          | Left-click → Class A · Right-click → Class B                |
| **Configurable architecture** | 0–4 hidden layers · 1–8 neurons per layer                   |
| **Activation functions**      | ReLU · Sigmoid · Tanh · Linear                              |
| **Live decision boundary**    | 110×110 grid updated in real time                           |
| **Weight/bias sliders**       | Manual control of every parameter                           |
| **Gradient descent**          | Full backprop with configurable LR & steps/frame            |
| **Network diagram**           | Neurons coloured by activation · connections by weight sign |
| **Loss chart**                | Running cross-entropy loss history                          |

---

## Project Structure

```
neural-net-viz/
├── index.html          ← entry point (open this)
├── style.css       ← dark cyberpunk lab theme
├── nn.js           ← pure-JS neural network (no dependencies)
├── app.js          ← canvas rendering & UI logic
└── README.md
```

---

## Quick Start (local)

```bash
# Option 1 — just open the file
open index.html

# Option 2 — serve with Python
python -m http.server 8080
# then visit http://localhost:8080

# Option 3 — serve with Node
npx serve .
```

---

## Deploy to GitHub Pages

1. Push this folder to a GitHub repository.
2. Go to **Settings → Pages**.
3. Under _Source_, select **Deploy from a branch** → `main` → `/ (root)`.
4. Save. Your app will be live at `https://<username>.github.io/<repo>/` within ~60 seconds.

---

## Deploy to Vercel

```bash
# Install Vercel CLI once
npm i -g vercel

# From the project folder
vercel

# Follow the prompts (all defaults work).
# Vercel detects a static site automatically.
```

Or drag-and-drop the folder at [vercel.com/new](https://vercel.com/new).

---

## Deploy to Netlify

```bash
# Install Netlify CLI once
npm i -g netlify-cli

# From the project folder
netlify deploy --prod --dir .
```

Or drag-and-drop the folder at [app.netlify.com/drop](https://app.netlify.com/drop).

---

## How to Use

### 1 — Add data

- **Left-click** on the plot canvas to place a **Class A** point (green).
- **Right-click** to place a **Class B** point (pink).

### 2 — Configure the network

- Adjust **Hidden Layers** and **Neurons / Layer** sliders.
- Pick an **Activation Function**.
- Click **⟳ Rebuild Network** to apply changes.

### 3 — Train

- Click **▶ Train** to run gradient descent and watch the decision boundary animate.
- Click **⏸ Pause** to stop.
- Use **Step ×1** to advance one epoch at a time.

### 4 — Explore weights manually

- Scroll down to the **Weights & Biases** panel.
- Drag any slider to modify a weight or bias directly and watch the boundary update instantly.
- Click **Randomize** to reinitialise all weights randomly.

---

## Neural Network Implementation

The NN is implemented from scratch in `js/nn.js`:

- **Architecture**: fully connected layers, arbitrary depth.
- **Initialisation**: He (Kaiming) for hidden layers, zero biases.
- **Hidden activations**: ReLU / Sigmoid / Tanh / Linear (user-selectable).
- **Output activation**: always Sigmoid → probability ∈ (0, 1).
- **Loss**: Binary Cross-Entropy.
- **Optimiser**: mini-batch gradient descent (full-batch by default).
- **Gradient clipping**: ±5 to prevent exploding gradients.

---

## Tips

- **XOR problem**: place points in two diagonal quadrants per class and use ≥ 1 hidden layer with ReLU or Tanh.
- **Linear separability**: for linearly separable data, a network with 0 hidden layers (logistic regression) is sufficient.
- **Overfitting**: add many points with noise, use a large network, and watch the boundary overfit.
- **High learning rate**: set LR > 0.5 to observe unstable training / oscillation.

---

## License

MIT — free to use, modify, and deploy.
