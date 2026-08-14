## 贝叶斯优化（终端运行与结果查看）

本项目建议使用：

- `bayesian_tuning_universal.py`：贝叶斯优化主入口（Optuna + SQLite）
- `automated_tuning.py`：训练脚本（支持 `--run_once` 供调参脚本调用）
- `view_optimization_results.py`：查看/监控 SQLite 里的 study
- `终端操作指南.md`：更完整的命令说明

### 1) 运行（推荐：先 1 次冒烟）

```bash
cd /home/wlwang/sq/CDFS-D2AFC-2026-main

python bayesian_tuning_universal.py \
  --config ./config/GF5-1105.py \
  --n_trials 1 \
  --max_episode 1 \
  --seeds 1236
```

### 2) 正式优化（例：10 次）

```bash
python bayesian_tuning_universal.py \
  --config ./config/GF5-1105.py \
  --n_trials 10 \
  --max_episode 2000 \
  --seeds 1236,1237
```

### 3) 后台运行 + 日志

```bash
nohup python bayesian_tuning_universal.py \
  --config ./config/GF5-1105.py \
  --n_trials 50 \
  --max_episode 2000 \
  --seeds 1236,1237 \
  > optimization.log 2>&1 &
```

### 4) 结果查看

```bash
# 查看最新一次优化结果（摘要 / 最优参数 / Top trials）
python view_optimization_results.py --latest

# 实时监控（默认每 30 秒刷新）
python view_optimization_results.py --latest --monitor
```

