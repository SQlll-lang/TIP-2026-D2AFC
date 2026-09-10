# CDFS-D²AFC

Official PyTorch implementation of **CDFS-D²AFC**: Dual-Level Domain Alignment meets Fine-Grained Contrastive Learning for Cross-Scene Few-Shot Hyperspectral Image Classification.

This repository provides the code for cross-scene few-shot hyperspectral image classification, including band mapping, **GLAI-Former** feature extraction, source/target prototype-based few-shot classification, dual-level domain alignment (D²A), and masked fine-grained contrastive learning (Masked FSCL).

---

## Method Overview

Given a labeled source domain (rich categories) and a target domain with only a few labeled samples per class, the model maps spectra from different sensors into a shared space, extracts spatial–spectral features with GLAI-Former, and is trained with the following joint loss:

```
L = L_fsl + λ_ctx · L_fscl + λ_wd · L_WD + λ_disc · L_disc
```

Default loss weights for the Indian Pines experiments match the paper: `λ_ctx = 2.0`, `λ_wd = 0.001`, `λ_disc = 1.0`.

| Module | Role | Main code |
|--------|------|-----------|
| Mapping | 1×1 convolution mapping source/target bands to a shared dim `d_map=100` | `model/mapping.py` |
| GLAI-Former | Spatial–spectral global/local attention feature extraction | `model/GLAI_Former.py` |
| FSL | Prototypical network + Euclidean-distance cross-entropy | `train-IP.py`, `utils/` |
| D²A | Sinkhorn Wasserstein distance + Cross-Transformer adversarial alignment | `model/loss.py`, `GLAI_Former.py` |
| Masked FSCL | Spatial random masking (`σ=0.8`) + ConTeXLoss (`τ=0.1`, `γ=0.7`) | `utils/data_augment.py`, `model/loss.py` |

At test time, features are min–max normalized and classified with **1-NN**.

---

## Requirements

- Python ≥ 3.8 (recommended: 3.9–3.11)
- CUDA GPU (training defaults to `config['gpu'] = 0`)
- Dependencies listed in [`requirements.txt`](requirements.txt)

```bash
# 1) Install PyTorch for your CUDA version: https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 2) Install remaining dependencies
pip install -r requirements.txt
```

`optuna` is required only by the Bayesian tuning scripts; it is optional for reproducing the main experiments.

---

## Dataset

This repository does **not** ship the raw data. Please download the datasets and place them as follows (paths can be changed in `config/IP.py`).

The provided configuration corresponds to **ZY1-04-24 → Indian Pines** (5-shot).

```
datasets/
├── ZY10424_imdb_76_7_7.pickle          # Source: pre-extracted 7×7 patches, 76 bands, 19 classes
└── IP/
    ├── indian_pines_corrected.mat      # Target imagery
    └── indian_pines_gt.mat             # Target labels
```

Indian Pines is available from [Hyperspectral Remote Sensing Scenes](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes). The source-domain pickle should follow the paper preprocessing (patch size 7×7). `utils/chikusei_imdb_128.py` provides a reference for building a similar source IMDB.

Update the paths in `config/IP.py` to match your local layout. `train-IP.py` joins paths with `os.path.join(data_path, source_data)`, so:

- `data_path`: dataset root directory
- `source_data` / `target_data` / `target_data_gt`: filenames relative to `data_path` (do not use absolute paths here)

Example:

```python
config['data_path'] = './datasets'
config['source_data'] = 'ZY10424_imdb_76_7_7.pickle'
config['target_data'] = 'IP/indian_pines_corrected.mat'
config['target_data_gt'] = 'IP/indian_pines_gt.mat'
config['gpu'] = 0
```

---

## Usage

Run from the project root:

```bash
python train-IP.py --config ./config/IP.py
```

By default, the experiment is repeated over 10 random seeds, and logs are written under `./logs/`. View TensorBoard with `tensorboard --logdir ./logs`.

Main hyperparameters (Indian Pines, consistent with the paper):

| Item | Value |
|------|-------|
| patch size | 7 |
| mapped dim / embedding | 100 / 128 |
| episode | 5000 |
| lr / weight decay | 1e-3 / 1e-4 |
| target labeled samples per class | 5 |
| FSCL temperature / γ | 0.1 / 0.7 |
| spatial mask ratio | 0.8 |

After training, OA, AA, Kappa, and F1 are reported (mean ± std over runs). Best checkpoints are saved under the corresponding log directory and can be used for visualization with `generate_tsne_from_model.py`.

### t-SNE (optional)

```bash
python generate_tsne_from_model.py --model_path <path_to_ckpt.pth> --config ./config/IP.py
```

### Bayesian tuning (optional)

Hyperparameter search scripts are under `BO-based tuning tool/`, with entry point `bayesian_tuning_universal.py`. See the docs in that folder for detailed commands.

---

## Project Structure

```
CDFS-D2AFC-main/
├── train-IP.py                 # Main train/test entry (IP experiments)
├── generate_tsne_from_model.py # t-SNE from a saved checkpoint
├── config/
│   └── IP.py                   # Data paths and training hyperparameters
├── model/
│   ├── mapping.py              # Spectral-dimension mapping
│   ├── GLAI_Former.py          # GLAI-Former / CrossTransformer / DomainDiscriminator
│   └── loss.py                 # Sinkhorn WD, ConTeXLoss, etc.
├── utils/
│   ├── dataloader.py           # Episode tasks and target-domain loading
│   ├── data_augment.py         # Spatial random masking
│   ├── utils.py                # Data I/O, metrics, logging
│   └── loss_function.py        # Extra losses (not used in the main experiments)
└── BO-based tuning tool/       # Optuna-based tuning utilities
```

---

## Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@ARTICLE{11664290,
  author={Sun, Qi and Wang, Wuli and Xin, Hongquan and Wang, Jianbu and Li, Wei and Ren, Guangbo and Fang, Leyuan},
  journal={IEEE Transactions on Image Processing}, 
  title={Dual-Level Domain Alignment Meets Fine-Grained Contrast: Advancing Cross-Scene Few-Shot Hyperspectral Image Classification}, 
  year={2026},
  volume={35},
  number={},
  pages={9256-9271},
  keywords={Modeling;Labeling;Wetlands;Modules (abstract algebra);Contrastive learning;Hyperspectral imaging;Image classification;Transformers;IP networks;Learning (artificial intelligence);HSI classification;cross-domain few-shot;dual-level domain adaptation;supervised contrastive learning},
  doi={10.1109/TIP.2026.3724715}}
```

---

## Acknowledgement

This implementation targets the cross-scene few-shot hyperspectral classification setting, where the source and target domains come from different sensors/scenes. Feel free to open an Issue for questions about data preprocessing and reproduction.
