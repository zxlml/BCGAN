<div align="center">

# 🌀 BCGAN：双层循环对抗网络的鲁棒流形拟合

<div>
&nbsp;<a href="README.md">🇬🇧 English</a> | <a href="README_zh.md">🇨🇳 简体中文</a>
</div>

[![Paper](https://img.shields.io/badge/📄-论文-red)](BCGAN.pdf)
[![License](https://img.shields.io/badge/⚡-MIT_License-blue)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/✅-22_项测试通过-brightgreen)](tests/test_bcgan.py)

论文 ***《Bilevel Manifold Fitting》*** 中 **双层循环生成对抗网络（BCGAN）** 的官方实现。

</div>

## 📢 新闻

- **[2025/09]**：官方实现首次发布（双层 SSGDA + 掩码 MFCGAN）。
- **[2025/09]**：双层求解器工程升级——精确预算掩码采样、逐掩码内层适配、PBCS 式迭代剪枝；新增完整单元测试套件。

## ✨ 亮点

BCGAN 针对流形学习的一个长期痛点：**环境数据被加性噪声、像素噪声或无信息维度/通道污染**。
它将鲁棒流形拟合建模为一个 *概率双层极小极大* 问题，并用单时间尺度的梯度上升-下降方法求解。

1. **🎯 双层掩码元学习** — 上层元学习器通过投影策略梯度估计（REINFORCE 得分函数 +
   控制变量方差缩减）学习维度掩码的 Bernoulli 松弛 `s ∈ [0,1]^p`，并将其投影到
   截单纯形 `C_s = {0 ≤ s ≤ 1, ‖s‖₁ ≤ C}`（论文算法 2，精确二分法）。

2. **🔁 掩码 MFCGAN 下层问题（Eq.(6)）** — 下层是极小极大问题：极小化
   `θ₁ = {FM, D_X, D_Y}`，极大化 `θ₂ = {G_X, G_Y}`，循环一致性正则*经过掩码*作用：
   `λ‖m ⊙ (x − b_x)‖₁`。流形拟合模块（FM）在对抗管线内部提供噪声鲁棒的局部参考投影。

3. **📐 精确预算掩码采样** — 掩码样本**以精确支撑集大小 `‖m‖₀ = C` 为条件**采样
   （对应论文的离散可行集），从根本上杜绝了全零掩码以及生成器“借道”隐变量携带噪声
   维度的 carving 捷径。

4. **⚙️ 逐掩码内层适配** — 每个候选掩码都通过下层模型的深拷贝并做若干轮 Adam 适配后
   再评估（即 PBCS "train-to-convergence" 内层循环在极小极大场景下的推广），消除了
   用单一在线 `θ` 评估所有掩码带来的系统性偏差。

5. **📉 面向收敛性的调度** — 一阶投影梯度 + 递减步长（η, γ₁, γ₂ ∝ 1/t），并借鉴
   PBCS 的三次缓动迭代剪枝：预算上限 `C_t: p → C_target` 在外层循环的 `ts` 与 `te`
   区间内平滑过渡。

6. **🧪 完整测试** — 22 项单元/功能测试，覆盖投影最优性、数据生成、FM 数值稳健性、
   梯度流通、REINFORCE 无偏性、掩码识别以及端到端双层冒烟运行。

## ⚙️ 主要结果

论文合成基准（第 5.2 节，Eq.(12)–(14)）：在环境输入（circle、involute）或隐变量
（torus，corrupted output）上追加噪声维度 `N(0,100)`。评价指标为到隐含流形的距离
（mean ± SD、95% 分位数、超过 3σ（σ = 0.01）的样本比例）。

| 数据集 | 污染方式 | 论文结果（BCGAN，表 2–3） |
| :--- | :--- | :--- |
| Circle   | 无               | 7.21e-4 |
| Circle   | 输入 (N(0,100))  | **7.49e-4** |
| Involute | 无               | 5.08e-3 |
| Involute | 输入 (N(0,100))  | **5.17e-3** |
| Torus    | 无               | 6.37e-3 |
| Torus    | 输出 (N(0,100))  | **6.44e-3** |

注意污染场景相对干净场景几乎不退化——学习到的掩码会自动过滤噪声维度
（例如 3 维污染 circle 中 `m = [1,1,0]`）。

## ⚡ 快速开始

### 1. 环境配置

```bash
conda create -n bcgan python=3.10
conda activate bcgan
pip install torch>=1.10.0 torchvision>=0.11.0 numpy>=1.19.0 pandas>=1.1.0 \
            Pillow>=8.0.0 scipy>=1.5.0 dominate>=2.4.0 wandb>=0.10.0 matplotlib
```

### 2. 运行单元 / 功能测试

```bash
python -m unittest tests.test_bcgan -v
```

### 3. 复现合成基准实验

```bash
# 输入污染（噪声环境维度），表 2
python run_synthetic.py --dataset circle   --corrupt input --plot
python run_synthetic.py --dataset involute --corrupt input --plot

# 输出污染（噪声隐变量维度），表 3
python run_synthetic.py --dataset torus    --corrupt output --plot
```

每次运行会执行双层优化（算法 1）、选出硬掩码、带掩码微调，并在 `results/bcgan/`
下保存 `metrics.csv`、`history.csv`、`scores.csv`、`model.pt` 与散点图。

### 4. 在自己的向量 / 图像数据上训练

```bash
python train.py --dataroot ./datasets/circle --name circle --gpu_ids -1
```

配置由 `options/base_options.py` 解析，详见下方[核心参数](#-核心参数run_syntheticpy)表。

## 🔧 核心参数（`run_synthetic.py`）

| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--dataset` | `str` | `circle` | 合成基准：`circle`、`involute`、`torus`。 |
| `--corrupt` | `str` | `input` | 噪声维度 `N(0,100)` 追加位置：`none`、`input`、`output`。 |
| `--n_noisy` | `int` | `1` | 噪声维度数量。 |
| `--T` | `int` | `12` | 外层双层迭代次数。 |
| `--n_inner` | `int` | `20` | 每次外层迭代的下层 SGDA/Adam 轮数。 |
| `--K` | `int` | `2` | 策略梯度使用的预算条件掩码样本数。 |
| `--eta` | `float` | `0.05` | 上层 `s` 的步长。 |
| `--adapt_steps` | `int` | `200` | 每个候选掩码的 `θ` 拷贝内层 Adam 适配轮数。 |
| `--adapt_lr` | `float` | `1e-3` | 逐掩码内层适配的学习率。 |
| `--meta_d_weight` | `float` | `0.1` | 上层反馈中对抗 D 项的权重。 |
| `--gamma1` / `--gamma2` | `float` | `1e-3` | 下层 `θ₁ = {FM, D}` 与 `θ₂ = {G}` 的步长。 |
| `--lambda_cycle` | `float` | `10.0` | 循环一致性权重 λ。 |
| `--C_target` | `float` | 自动 | 最终掩码预算 `‖m‖₀`（自动：`p − n_noisy`）。 |
| `--ts` / `--te` | `float` | `0.3` / `0.7` | 三次缓动剪枝窗口（占 `T` 的比例）。 |
| `--finetune` | `int` | `6000` | 硬掩码微调步数（恒定 + 余弦衰减）。 |
| `--lower_opt` | `str` | `adam` | 下层优化器（算法 1 的 `sgd`，或 GAN betas 的 `adam`）。 |
| `--sigma` | `float` | `0.01` | 合成数据的输入噪声标准差 σ。 |
| `--plot` | 开关 | 关 | 保存输入 / 去噪 / 生成点的散点图。 |

## 📁 项目结构

```
BCGAN/
├── bcgan_core.py        # 双层核心：截单纯形投影、精确预算掩码采样、
│                        #   掩码 MFCGAN 下层、SSGDA 求解器
├── run_synthetic.py     # 论文合成基准（表 2-3）+ 指标评估
├── train.py             # 向量 / 图像数据训练入口
├── models/              # 网络（FFN 生成器/判别器）、MF 流形拟合模块、
│                        #   CycleGAN/MFCGAN 模型定义
├── datasets/            # 合成数据生成器（circle / involute / torus）+ csv 数据
├── data/                # 数据集加载器（vector、vec2pic、MFpic）
├── options/             # 命令行参数（BaseOptions / TrainOptions）
├── util/                # 可视化、图像池、html 日志等
├── hypergrad/           # 超梯度工具
├── coreset_utils/       # 核集选择工具（PBCS）
├── reinforce_utils/     # REINFORCE / 策略梯度工具
├── logging_utils/       # 目录管理、tensorboard 工具
├── tests/               # 单元与功能测试套件
├── BCGAN.pdf            # 论文
└── LICENSE              # MIT
```

## 🚧 TODO

- [ ] 真实数据基准（含噪声通道的图像去噪）
- [ ] 多 GPU 训练支持
- [ ] 预训练模型权重

## 🤝 致谢

本项目建立在 [MFCGAN](https://github.com/zhigang-yao/MFCGAN)（流形拟合 CycleGAN）
的优秀工程之上，并借鉴了
[PBCS](https://github.com/qichaosustech/Probabilistic-Bilevel-Coreset-Selection)
的一阶投影梯度双层设计。CycleGAN 代码结构遵循
[CycleGAN/pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)。

## ⭐️ 引用

如果 BCGAN 对您的研究有帮助，请考虑引用：

```bibtex
@article{bcgan2025,
  title   = {Bilevel Manifold Fitting},
  author  = {BCGAN Authors},
  year    = {2025},
  url     = {https://github.com/zxlml/BCGAN}
}
```

## 📄 许可证

本项目基于 [MIT License](LICENSE) 开源。

<div align="center">

**如果这个仓库对您有帮助，请给一个 ⭐！**

</div>
