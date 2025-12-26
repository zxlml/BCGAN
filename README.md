# Bilevel Manifold Fitting

### Bilevel Cycle GAN for Robust Manifold Fitting

This repository contains the official implementation of Bilevel Cycle Generative Adversarial Network (BCGAN), as presented in the paper "Bilevel Manifold Fitting".


## Overview

Manifold learning assumes that high-dimensional ambient data possess a low-dimensional geometric structure. However, existing methods are often vulnerable to corrupted data, particularly noisy dimensions or channels. This project proposes a bilevel probabilistic optimization strategy to address these issues.

## Key Features

Bilevel Optimization: Utilizes an upper-level meta-learner to automatically learn masks for features/channels, filtering out noisy dimensions.

Robust Manifold Fitting: Integrates a CycleGAN framework with a manifold fitting module to learn robust mutual mappings between ambient space (N) and latent space (M).

Theoretical Guarantees: Provides generalization bounds for stochastic bilevel minimax problems via algorithmic stability.

Versatile Applications: Supports manifold reconstruction, noise reduction, synthetic sample generation, and nonlinear interpolation.



## Environment Requirements

This project is developed using Python and PyTorch. Ensure you have the following environment installed:

`torch>=1.10.0`
`torchvision>=0.11.0`
`numpy>=1.19.0`
`pandas>=1.1.0`
`Pillow>=8.0.0`
`scipy>=1.5.0`
`dominate>=2.4.0`
`wandb>=0.10.0`


## Core Parameters

The configuration of the experiments is handled via the BaseOptions class, which parses command-line arguments. Below is a detailed explanation of the core parameters provided in the implementation.


| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dataroot` | `str` | `'./datasets/circle/'` | Root path to the dataset. Should contain training and validation subsets. |
| `--name` | `str` | `'circle'` | Name of the experiment. Determines where to store the generated models and samples. |
| `--gpu_ids` | `str` | `'-1'` | Specifies GPU IDs. Use comma-separated values (e.g., `0,1`) for multi-GPU training. Use `-1` for CPU mode. |
| `--checkpoints_dir`| `str` | `'./checkpoints'` | Directory where model checkpoints are saved. |
| `--res_dir` | `str` | `'./results'` | Directory where visualization results and metrics are saved. |
| `--K` | `int` | `'20'` | Size of coreset. |

