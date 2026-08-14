"""
通用贝叶斯超参数优化（终端运行版）

思路：
 - Optuna 负责采样超参数
 - 每个 trial 通过 subprocess 调用 `automated_tuning.py --run_once ...`
 - 从 stdout 里解析 `TUNE_RESULT_JSON=...`，取 best_accuracy 作为 objective

优点：不需要把训练代码再复制一份到调参脚本里；训练逻辑只维护在 `automated_tuning.py`。
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime

import optuna


def load_config(config_path: str):
    spec = importlib.util.spec_from_file_location("cfg", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载配置文件: {config_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod.config


def detect_dataset_name(cfg) -> str:
    target = str(cfg.get("target_data", ""))
    target_upper = target.upper()
    # 常见数据集关键字
    for key in ["GF51105", "GF50525", "GF5_YANCHENG", "GF5_YC", "HOUSTON", "IP", "INDIAN_PINES", "ZY10424", "ZY1-0424"]:
        if key in target_upper:
            if key in ["GF5_YANCHENG", "GF5_YC"]:
                return "GF5YC"
            if key in ["INDIAN_PINES"]:
                return "IP"
            return key.replace("-", "")
    # 兜底：取父目录名
    base = os.path.basename(os.path.dirname(target))
    return base if base else "DATASET"


def run_one_trial(args, params) -> float:
    """
    调用 automated_tuning.py 跑一次，返回 best_accuracy（越大越好）
    """
    cmd = [
        sys.executable,
        "automated_tuning.py",
        "--config",
        args.config,
        "--run_once",
        "--lambda_weight",
        str(params["lambda_weight"]),
        "--mask_ratio",
        str(params["mask_ratio"]),
        "--weight_ctx",
        str(params["weight_ctx"]),
        "--weight_wd",
        str(params["weight_wd"]),
        "--weight_disc",
        str(params["weight_disc"]),
    ]
    if args.max_episode is not None:
        cmd += ["--max_episode", str(args.max_episode)]
    if args.seeds is not None and str(args.seeds).strip() != "":
        cmd += ["--seeds", str(args.seeds)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    if proc.returncode != 0:
        # 打印一点关键信息，便于你在终端里定位
        print("[trial subprocess] returncode != 0")
        print(stderr[-4000:])
        return 0.0

    # 解析机器可读输出
    best_acc = None
    for line in stdout.splitlines():
        if line.startswith("TUNE_RESULT_JSON="):
            payload = json.loads(line.split("=", 1)[1])
            best_acc = float(payload["best_accuracy"])
            break

    if best_acc is None:
        print("[trial subprocess] 未找到 TUNE_RESULT_JSON 行")
        print(stdout[-2000:])
        return 0.0

    return best_acc


def main():
    parser = argparse.ArgumentParser(description="Bayesian hyperparameter tuning (universal)")
    parser.add_argument("--config", type=str, default=os.path.join("./config", "GF5-1105.py"))
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--max_episode", type=int, default=2000, help="每个 trial 最大训练轮次（加速调参）")
    parser.add_argument("--seeds", type=str, default="1236,1237", help="每个 trial 的 seeds（逗号分隔）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_name = detect_dataset_name(cfg)

    study_name = f"{dataset_name}_bayesian_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    storage = f"sqlite:///{study_name}.db"

    print("=" * 80)
    print(f"贝叶斯超参数优化 - {dataset_name} 数据集")
    print("=" * 80)
    print(f"Study名称: {study_name}")
    print(f"数据库路径: {storage}")
    print(f"试验次数: {args.n_trials}")
    print(f"超时时间: {args.timeout if args.timeout else '无限制'}")
    print(f"每 trial max_episode: {args.max_episode}")
    print(f"每 trial seeds: {args.seeds}")
    print("=" * 80)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=False,
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "lambda_weight": trial.suggest_float("lambda_weight", 0.0, 0.5, step=0.05),
            "mask_ratio": trial.suggest_float("mask_ratio", 0.5, 0.95, step=0.05),
            "weight_ctx": trial.suggest_float("weight_ctx", 0.5, 5.0, step=0.5),
            "weight_wd": trial.suggest_float("weight_wd", 1e-4, 1e-2, log=True),
            "weight_disc": trial.suggest_float("weight_disc", 0.1, 5.0, step=0.1),
        }
        print(f"\n{'='*80}\nTrial {trial.number}\nparams={params}\n{'='*80}\n")
        return run_one_trial(args, params)

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, show_progress_bar=True)

    results = {
        "dataset": dataset_name,
        "study_name": study_name,
        "storage": storage,
        "n_trials": len(study.trials),
        "best_trial": study.best_trial.number if study.best_trial else None,
        "best_accuracy": float(study.best_value) if study.best_value is not None else None,
        "best_params": study.best_params if study.best_trial else None,
        "timestamp": datetime.now().isoformat(),
    }

    out_json = f"{study_name}_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("优化完成")
    print("=" * 80)
    print(f"best_accuracy: {results['best_accuracy']}")
    print(f"best_params: {results['best_params']}")
    print(f"结果已保存: {out_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()

