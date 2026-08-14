# CDFS-D²AFC

Official PyTorch implementation of **CDFS-D²AFC**: Dual-Level Domain Alignment with Fine-Grained Contrastive Learning for Cross-Scene Few-Shot Hyperspectral Image Classification.

跨场景高光谱少样本分类代码。本仓库实现论文方法：波段映射 + **GLAI-Former** 特征提取、源/目标域原型少样本分类、双层域对齐（D²A）以及掩码细粒度对比学习（Masked FSCL）。

---

## Method Overview

给定源域（类别丰富、有标签）和目标域（每类仅少量标注），模型将不同传感器的光谱维映射到公共空间，再用 GLAI-Former 提取空–谱特征，并通过下列损失联合训练：

```
L = L_fsl + λ_ctx · L_fscl + λ_wd · L_WD + λ_disc · L_disc
```

Indian Pines 实验默认权重与论文一致：`λ_ctx = 2.0`，`λ_wd = 0.001`，`λ_disc = 1.0`。

| 模块 | 作用 | 主要代码 |
|------|------|----------|
| Mapping | 1×1 卷积将源/目标波段映到公共维 `d_map=100` | `model/mapping.py` |
| GLAI-Former | 空–谱全局/局部注意力特征提取 | `model/GLAI_Former.py` |
| FSL | 原型网络 + 欧氏距离交叉熵 | `train-IP.py`, `utils/` |
| D²A | Sinkhorn Wasserstein 距离 + Cross-Transformer 对抗对齐 | `model/loss.py`, `GLAI_Former.py` |
| Masked FSCL | 空间随机掩码 (`σ=0.8`) + ConTeXLoss (`τ=0.1`, `γ=0.7`) | `utils/data_augment.py`, `model/loss.py` |

测试阶段对特征做 min-max 归一化后使用 **1-NN** 分类。

---

## Requirements

- Python ≥ 3.8（建议 3.9–3.11）
- CUDA GPU（训练默认 `config['gpu'] = 0`）
- 依赖见 [`requirements.txt`](requirements.txt)

```bash
# 1) 按本机 CUDA 版本安装 PyTorch：https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 2) 其余依赖
pip install -r requirements.txt
```

`optuna` 仅在贝叶斯调参脚本中需要；只复现主实验可不安装。

---

## Dataset

本仓库**不包含**原始数据。请自行下载并按下面结构放置（路径可在 `config/IP.py` 中修改）。

当前提供的配置对应：**ZY1-04-24 → Indian Pines**（5-shot）。

```
datasets/
├── ZY10424_imdb_76_7_7.pickle          # 源域：已切好的 7×7 patch，76 波段，19 类
└── IP/
    ├── indian_pines_corrected.mat      # 目标域影像
    └── indian_pines_gt.mat             # 目标域标签
```

Indian Pines 可从 [Hyperspectral Remote Sensing Scenes](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) 获取。源域 pickle 需按论文预处理（patch 大小 7×7）。`utils/chikusei_imdb_128.py` 给出了类似的源域 IMDB 构建参考。

在 `config/IP.py` 中把路径改成你的本地目录。`train-IP.py` 会用 `os.path.join(data_path, source_data)` 拼接路径，因此：

- `data_path`：数据集根目录
- `source_data` / `target_data` / `target_data_gt`：相对 `data_path` 的文件名（不要再写成绝对路径）

示例：

```python
config['data_path'] = './datasets'
config['source_data'] = 'ZY10424_imdb_76_7_7.pickle'
config['target_data'] = 'IP/indian_pines_corrected.mat'
config['target_data_gt'] = 'IP/indian_pines_gt.mat'
config['gpu'] = 0
```

---

## Usage

在项目根目录运行：

```bash
python train-IP.py --config ./config/IP.py
```

默认会按 10 组随机种子重复实验，并在 `./logs/` 下写入日志。TensorBoard 记录可通过 `tensorboard --logdir ./logs` 查看。

主要超参（Indian Pines，与论文一致）：

| 项 | 值 |
|----|----|
| patch size | 7 |
| mapped dim / embedding | 100 / 128 |
| episode | 5000 |
| lr / weight decay | 1e-3 / 1e-4 |
| target labeled samples per class | 5 |
| FSCL temperature / γ | 0.1 / 0.7 |
| spatial mask ratio | 0.8 |

训练结束后会报告 OA、AA、Kappa、F1（多次运行的均值与标准差）。最佳权重会保存在日志对应目录中，可用 `generate_tsne_from_model.py` 做特征可视化。

### t-SNE（可选）

```bash
python generate_tsne_from_model.py --model_path <path_to_ckpt.pth> --config ./config/IP.py
```

### Bayesian tuning（可选）

超参搜索脚本在 `BO-based tuning tool/`，入口为 `bayesian_tuning_universal.py`。详细命令见该目录下的说明文档。

---

## Project Structure

```
CDFS-D2AFC-main/
├── train-IP.py                 # 主训练 / 测试入口（IP 实验）
├── generate_tsne_from_model.py # 用已保存权重生成 t-SNE
├── config/
│   └── IP.py                   # 数据路径与训练超参
├── model/
│   ├── mapping.py              # 光谱维映射
│   ├── GLAI_Former.py          # GLAI-Former / CrossTransformer / DomainDiscriminator
│   └── loss.py                 # Sinkhorn WD、ConTeXLoss 等
├── utils/
│   ├── dataloader.py           # episode 任务与目标域加载
│   ├── data_augment.py         # 空间随机掩码
│   ├── utils.py                # 数据读取、度量、日志
│   └── loss_function.py        # 额外损失（主实验未使用）
└── BO-based tuning tool/       # Optuna 调参工具
```

---

## Citation

如果本代码对你的研究有帮助，请引用我们的论文：

```bibtex
@article{CDFS-D2AFC,
  title   = {CDFS-D$^{2}$AFC: Dual-Level Domain Alignment with Fine-Grained Contrastive Learning for Cross-Scene Few-Shot Hyperspectral Image Classification},
  author  = {},
  journal = {IEEE Transactions on Image Processing},
  year    = {2026}
}
```

论文正式发表后，请将上面的 `author`、卷期页码和 DOI 补全。

---

## Acknowledgement

本实现面向跨场景高光谱少样本分类设定，源域与目标域来自不同传感器/场景。欢迎提 Issue 讨论数据预处理与复现细节。
