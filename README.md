<div align="center">

# 🌀 BCGAN: Bilevel CycleGAN for Robust Manifold Fitting

<div>
&nbsp;<a href="README.md">🇬🇧 English</a> | <a href="README_zh.md">🇨🇳 简体中文</a>
</div>

[![Paper](https://img.shields.io/badge/📄-Paper-red)](BCGAN.pdf)
[![License](https://img.shields.io/badge/⚡-MIT_License-blue)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/✅-22_tests_passing-brightgreen)](tests/test_bcgan.py)

Official implementation of **Bilevel Cycle Generative Adversarial Network (BCGAN)**,
as presented in the paper ***"Bilevel Manifold Fitting"***.

</div>

## 📢 News

- **[2025/09]**: Initial release of the official implementation (bilevel SSGDA + masked MFCGAN).
- **[2025/09]**: Engineering upgrade of the bilevel solver — exact-budget mask sampling, per-mask inner adaptation, and PBCS-style iterative pruning; full unit-test suite added.

## ✨ Highlights

BCGAN addresses a long-standing weakness of manifold learning: **ambient data corrupted
by additive noise, pixel noise, or uninformative dimensions/channels**. It casts robust
manifold fitting as a *probabilistic bilevel minimax* problem and solves it with a
single-time-scale gradient ascent–descent method.

1. **🎯 Bilevel Mask Meta-Learning** — An upper-level meta-learner learns a Bernoulli
   relaxation `s ∈ [0,1]^p` of the dimension mask via projected policy-gradient
   estimation (REINFORCE score function + control-variate variance reduction), and
   projects it onto the capped simplex `C_s = {0 ≤ s ≤ 1, ‖s‖₁ ≤ C}` (Algorithm 2,
   exact bisection).

2. **🔁 Masked MFCGAN Lower Level (Eq.(6))** — The lower level is a minimax problem:
   `θ₁ = {FM, D_X, D_Y}` is minimized while `θ₂ = {G_X, G_Y}` is maximized, with
   cycle-consistency regularization applied *through the mask*
   `λ‖m ⊙ (x − b_x)‖₁`. A manifold fitting module (FM) provides noise-robust
   local reference projections inside the adversarial pipeline.

3. **📐 Exact-Budget Mask Sampling** — Mask samples are drawn **conditioned on the
   exact support size `‖m‖₀ = C`** (the paper's discrete feasible set), which rules out
   degenerate all-zero masks and the "carving" shortcut, where the generator re-uses
   the latent code to carry noisy dimensions.

4. **⚙️ Per-Mask Inner Adaptation** — Every candidate mask is evaluated through a deep
   copy of the lower model adapted for a few Adam rounds (the "train-to-convergence"
   inner loop of PBCS, extended to the minimax setting), removing the systematic bias
   of evaluating all masks through a single online `θ`.

5. **📉 Convergence-Oriented Schedules** — First-order projected gradient with decaying
   stepsizes (η, γ₁, γ₂ ∝ 1/t), plus PBCS-style cubic iterative pruning of the budget
   cap `C_t: p → C_target` between fractions `ts` and `te` of the outer loop.

6. **🧪 Tested** — 22 unit/functional tests covering the projection (optimality-checked),
   data generation, FM numerical robustness, gradient flow, REINFORCE unbiasedness,
   mask identification, and end-to-end bilevel smoke runs.

## ⚙️ Main Results

Synthetic benchmarks of the paper (Section 5.2, Eq.(12)–(14)): a noisy dimension
`N(0,100)` is appended to the ambient input (circle, involute) or to the latent code
(torus, corrupted output). The metric is the distance to the latent manifold
(mean ± SD, 95% quantile, and the percentage of samples beyond 3σ, σ = 0.01).

| Dataset | Corruption | Paper (BCGAN, Tables 2–3) |
| :--- | :--- | :--- |
| Circle   | none            | 7.21e-4 |
| Circle   | input  (N(0,100)) | **7.49e-4** |
| Involute | none            | 5.08e-3 |
| Involute | input  (N(0,100)) | **5.17e-3** |
| Torus    | none            | 6.37e-3 |
| Torus    | output (N(0,100)) | **6.44e-3** |

Note how the corrupted results barely degrade w.r.t. the clean ones — the learned mask
filters the noisy dimension automatically (e.g. `m = [1,1,0]` for the 3-D corrupted circle).

## ⚡ Quick Start

### 1. Environment

```bash
conda create -n bcgan python=3.10
conda activate bcgan
pip install torch>=1.10.0 torchvision>=0.11.0 numpy>=1.19.0 pandas>=1.1.0 \
            Pillow>=8.0.0 scipy>=1.5.0 dominate>=2.4.0 wandb>=0.10.0 matplotlib
```

### 2. Run the unit / functional test suite

```bash
python -m unittest tests.test_bcgan -v
```

### 3. Reproduce the synthetic benchmarks

```bash
# corrupted input (noisy ambient dimension), Table 2
python run_synthetic.py --dataset circle   --corrupt input --plot
python run_synthetic.py --dataset involute --corrupt input --plot

# corrupted output (noisy latent dimension), Table 3
python run_synthetic.py --dataset torus    --corrupt output --plot
```

Each run performs the bilevel optimization (Algorithm 1), selects the hard mask,
fine-tunes with the mask, and saves `metrics.csv`, `history.csv`, `scores.csv`,
`model.pt` and a scatter plot under `results/bcgan/`.

### 4. Training on your own vector / image data

```bash
python train.py --dataroot ./datasets/circle --name circle --gpu_ids -1
```

The configuration is parsed by `options/base_options.py`; see the
[Core Parameters](#-core-parameters) table below.

## 🔧 Core Parameters (`run_synthetic.py`)

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dataset` | `str` | `circle` | Synthetic benchmark: `circle`, `involute`, `torus`. |
| `--corrupt` | `str` | `input` | Where the noisy dimension `N(0,100)` is appended: `none`, `input`, `output`. |
| `--n_noisy` | `int` | `1` | Number of noisy dimensions. |
| `--T` | `int` | `12` | Outer bilevel iterations. |
| `--n_inner` | `int` | `20` | Lower SGDA/Adam rounds per outer iteration. |
| `--K` | `int` | `2` | Budget-conditioned mask samples for the policy gradient. |
| `--eta` | `float` | `0.05` | Upper-level step size for `s`. |
| `--adapt_steps` | `int` | `200` | Inner Adam rounds adapting a copy of `θ` per candidate mask. |
| `--adapt_lr` | `float` | `1e-3` | Learning rate of the per-mask inner adaptation. |
| `--meta_d_weight` | `float` | `0.1` | Weight of the adversarial D-terms in the upper feedback. |
| `--gamma1` / `--gamma2` | `float` | `1e-3` | Lower step sizes for `θ₁ = {FM, D}` and `θ₂ = {G}`. |
| `--lambda_cycle` | `float` | `10.0` | Cycle-consistency weight λ. |
| `--C_target` | `float` | auto | Final mask budget `‖m‖₀` (auto: `p − n_noisy`). |
| `--ts` / `--te` | `float` | `0.3` / `0.7` | Pruning window (fractions of `T`) of the cubic cap schedule. |
| `--finetune` | `int` | `6000` | Fine-tuning steps with the hard mask (constant + cosine decay). |
| `--lower_opt` | `str` | `adam` | Lower-level optimizer (`sgd` per Algorithm 1, or `adam` with GAN betas). |
| `--sigma` | `float` | `0.01` | Input noise std σ of the synthetic data. |
| `--plot` | flag | off | Save scatter plots of input / denoised / generated points. |

## 📁 Project Structure

```
BCGAN/
├── bcgan_core.py        # bilevel core: capped-simplex projection, exact-budget
│                        #   mask sampling, masked MFCGAN lower level, SSGDA solver
├── run_synthetic.py     # paper's synthetic benchmarks (Tables 2-3) + metrics
├── train.py             # training entry for vector / image data
├── models/              # networks (FFN generator/discriminator), MF manifold
│                        #   fitting module, CycleGAN/MFCGAN model definitions
├── datasets/            # synthetic generators (circle / involute / torus) + csv data
├── data/                # dataset loaders (vector, vec2pic, MFpic)
├── options/             # command-line options (BaseOptions / TrainOptions)
├── util/                # visualizer, image pool, html logger, ...
├── hypergrad/           # hypergradient utilities
├── coreset_utils/       # coreset selection utilities (PBCS)
├── reinforce_utils/     # REINFORCE / policy-gradient utilities
├── logging_utils/       # directory management, tensorboard tools
├── tests/               # unit & functional test suite
├── BCGAN.pdf            # the paper
└── LICENSE              # MIT
```

## 🚧 TODO

- [ ] Real-data benchmarks (image denoising with noisy channels)
- [ ] Multi-GPU training support
- [ ] Pretrained checkpoints

## 🤝 Acknowledgements

This project builds upon the excellent engineering of
[MFCGAN](https://github.com/zhigang-yao/MFCGAN) (manifold-fitting CycleGAN) and
reuses ideas from the first-order projected-gradient bilevel design of
[PBCS](https://github.com/qichaosustech/Probabilistic-Bilevel-Coreset-Selection).
The CycleGAN codebase follows the
[CycleGAN/pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) structure.

## ⭐️ Citation

If you find BCGAN useful for your research, please consider citing:

```bibtex
@article{bcgan2025,
  title   = {Bilevel Manifold Fitting},
  author  = {BCGAN Authors},
  year    = {2025},
  url     = {https://github.com/zxlml/BCGAN}
}
```

## 📄 License

This project is released under the [MIT License](LICENSE).

<div align="center">

**If this repository helps you, please give it a ⭐!**

</div>
