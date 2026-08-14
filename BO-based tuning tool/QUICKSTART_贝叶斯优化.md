## Quickstart（只含终端命令）

### 0) 进入目录

```bash
cd /home/wlwang/sq/CDFS-D2AFC-2026-main
```

### 1) 环境自检

```bash
python test_bayesian_setup.py
```

### 2) 先跑 1 次冒烟（强烈推荐）

```bash
python bayesian_tuning_universal.py \
  --config ./config/GF5-1105.py \
  --n_trials 1 \
  --max_episode 1 \
  --seeds 1236
```

### 3) 正式优化（示例 10 次）

```bash
python bayesian_tuning_universal.py \
  --config ./config/GF5-1105.py \
  --n_trials 10 \
  --max_episode 2000 \
  --seeds 1236,1237
```

### 4) 看结果 / 监控

```bash
python view_optimization_results.py --latest
python view_optimization_results.py --latest --monitor
```

