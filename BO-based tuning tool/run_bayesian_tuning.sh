#!/bin/bash

# GF51105 贝叶斯超参数优化启动脚本

echo "=========================================="
echo "GF51105 贝叶斯超参数优化"
echo "=========================================="
echo ""

# 设置默认参数
N_TRIALS=50
TIMEOUT=""
CONFIG="./config/GF5-1105.py"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --n_trials)
            N_TRIALS="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="--timeout $2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --help|-h)
            echo "用法: ./run_bayesian_tuning.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --n_trials NUM    试验次数 (默认: 50)"
            echo "  --timeout SEC     超时时间/秒 (默认: 无限制)"
            echo "  --config PATH     配置文件路径 (默认: ./config/GF5-1105.py)"
            echo "  --help, -h        显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  ./run_bayesian_tuning.sh                              # 默认参数"
            echo "  ./run_bayesian_tuning.sh --n_trials 100               # 100次试验"
            echo "  ./run_bayesian_tuning.sh --n_trials 50 --timeout 7200 # 2小时超时"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

echo "配置:"
echo "  试验次数: $N_TRIALS"
echo "  超时时间: ${TIMEOUT:-无限制}"
echo "  配置文件: $CONFIG"
echo ""

# 检查必要的文件
if [ ! -f "$CONFIG" ]; then
    echo "错误: 配置文件不存在: $CONFIG"
    exit 1
fi

if [ ! -f "bayesian_tuning_universal.py" ]; then
    echo "错误: 找不到 bayesian_tuning_universal.py"
    exit 1
fi

# 检查Python依赖
echo "检查依赖..."
python -c "import optuna" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "错误: Optuna 未安装"
    echo "请运行: pip install optuna"
    exit 1
fi

python -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "错误: PyTorch 未安装"
    exit 1
fi

echo "依赖检查完成"
echo ""

# 创建日志目录
mkdir -p logs_bayesian
LOG_FILE="logs_bayesian/optimization_$(date +%Y%m%d_%H%M%S).log"

echo "开始优化..."
echo "日志文件: $LOG_FILE"
echo ""
echo "=========================================="
echo ""

# 运行优化
python bayesian_tuning_universal.py \
    --config "$CONFIG" \
    --n_trials $N_TRIALS \
    $TIMEOUT \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=$?

echo ""
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "优化完成!"
    echo ""
    echo "结果文件:"
    echo "  - JSON结果: GF51105_bayesian_opt_*_results.json"
    echo "  - SQLite数据库: GF51105_bayesian_opt_*.db"
    echo "  - 优化历史图: GF51105_bayesian_opt_*_history.png"
    echo "  - 参数重要性图: GF51105_bayesian_opt_*_importance.png"
    echo "  - 日志文件: $LOG_FILE"
    echo ""
    echo "下一步:"
    echo "  1. 查看 *_results.json 获取最佳超参数"
    echo "  2. 运行 python apply_best_params.py 应用最佳参数"
    echo "  3. 使用最佳参数重新训练完整模型"
else
    echo "优化失败 (退出码: $EXIT_CODE)"
    echo "请查看日志: $LOG_FILE"
fi

echo "=========================================="
