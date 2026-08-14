"""
测试贝叶斯优化环境配置
运行此脚本以验证所有依赖和文件是否正确
"""

import sys
import os
import importlib.util


def load_config(config_path):
    spec = importlib.util.spec_from_file_location("exp_config", config_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.config


def test_dependencies():
    """测试Python依赖"""
    print("="*80)
    print("测试 Python 依赖")
    print("="*80)
    print(f"Python解释器: {sys.executable}")
    print(f"Python版本: {sys.version.split()[0]}")
    
    required_packages = {
        'numpy': 'NumPy',
        'torch': 'PyTorch',
        'sklearn': 'scikit-learn',
        'optuna': 'Optuna',
        'pickle': 'pickle',
        'tensorboardX': 'TensorboardX'
    }
    
    missing = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name:<20} 已安装")
        except Exception as e:
            print(f"✗ {name:<20} 不可用 ({type(e).__name__}: {e})")
            missing.append(name)
    
    if missing:
        print(f"\n缺少依赖: {', '.join(missing)}")
        print("请运行: pip install " + " ".join(missing))
        return False
    
    print("\n✓ 所有依赖已安装")
    return True


def test_files():
    """测试必要文件是否存在"""
    print("\n" + "="*80)
    print("测试文件完整性")
    print("="*80)
    
    required_files = {
        'bayesian_tuning_wrapper.py': '贝叶斯优化主程序',
        'apply_best_params.py': '应用参数工具',
        'view_optimization_results.py': '查看结果工具',
        'run_bayesian_tuning.sh': '启动脚本',
        'config/GF5-1105.py': '配置文件',
        'README_BAYESIAN_TUNING.md': '使用说明',
        'QUICKSTART_贝叶斯优化.md': '快速入门'
    }
    
    missing = []
    
    for file, desc in required_files.items():
        if os.path.exists(file):
            print(f"✓ {desc:<20} {file}")
        else:
            print(f"✗ {desc:<20} {file} (缺失)")
            missing.append(file)
    
    if missing:
        print(f"\n缺少文件: {', '.join(missing)}")
        return False
    
    print("\n✓ 所有文件完整")
    return True


def test_config():
    """测试配置文件加载"""
    print("\n" + "="*80)
    print("测试配置文件")
    print("="*80)
    
    try:
        config_path = './config/GF5-1105.py'
        config = load_config(config_path)
        
        train_opt = config['train_config']
        
        print(f"✓ 配置文件加载成功")
        print(f"\n关键配置:")
        print(f"  数据集: {config['target_data']}")
        print(f"  类别数: {train_opt['tar_class_num']}")
        print(f"  每类样本数: {train_opt['tar_lsample_num_per_class']}")
        print(f"  训练轮次: {train_opt['episode']}")
        print(f"  GPU: {config['gpu']}")
        
        return True
        
    except Exception as e:
        print(f"✗ 配置文件加载失败: {str(e)}")
        return False


def test_data_paths():
    """测试数据路径"""
    print("\n" + "="*80)
    print("测试数据路径")
    print("="*80)
    
    try:
        config = load_config('./config/GF5-1105.py')
        
        data_path = config['data_path']
        source_data = config['source_data']
        target_data = config['target_data']
        target_data_gt = config['target_data_gt']
        
        # 检查源域数据
        source_file = os.path.join(data_path, source_data)
        if os.path.exists(source_file):
            print(f"✓ 源域数据: {source_file}")
        else:
            print(f"✗ 源域数据不存在: {source_file}")
        
        # 检查目标域数据
        target_file = os.path.join(data_path, target_data)
        if os.path.exists(target_file):
            print(f"✓ 目标域数据: {target_file}")
        else:
            print(f"✗ 目标域数据不存在: {target_file}")
        
        # 检查标签数据
        target_gt_file = os.path.join(data_path, target_data_gt)
        if os.path.exists(target_gt_file):
            print(f"✓ 标签数据: {target_gt_file}")
        else:
            print(f"✗ 标签数据不存在: {target_gt_file}")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据路径检查失败: {str(e)}")
        return False


def test_gpu():
    """测试GPU可用性"""
    print("\n" + "="*80)
    print("测试 GPU")
    print("="*80)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✓ CUDA 可用")
            print(f"  GPU数量: {gpu_count}")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # 测试GPU计算
            x = torch.rand(100, 100).cuda()
            y = torch.rand(100, 100).cuda()
            z = torch.matmul(x, y)
            print(f"\n✓ GPU 计算测试通过")
            
            return True
        else:
            print(f"✗ CUDA 不可用")
            print(f"  将使用 CPU（速度会很慢）")
            return False
            
    except Exception as e:
        print(f"✗ GPU 测试失败: {str(e)}")
        return False


def test_model_import():
    """测试模型导入"""
    print("\n" + "="*80)
    print("测试模型导入")
    print("="*80)
    
    try:
        from model.mapping import Mapping
        print(f"✓ Mapping 导入成功")
        
        from model.GLAI_Former import GLAIFormer, CrossTransformer, DomainDiscriminator
        print(f"✓ GLAI_Former 模块导入成功")
        
        from utils import utils, data_augment
        print(f"✓ Utils 模块导入成功")
        
        from model.loss import ConTeXLoss, SinkhornDistance
        print(f"✓ ConTeXLoss / SinkhornDistance 导入成功")
        
        print(f"\n✓ 所有模型模块导入成功")
        return True
        
    except Exception as e:
        print(f"✗ 模型导入失败: {str(e)}")
        print(f"  请确保在项目根目录运行此脚本")
        return False


def print_summary(results):
    """打印测试摘要"""
    print("\n" + "="*80)
    print("测试摘要")
    print("="*80)
    
    total = len(results)
    passed = sum(results.values())
    failed = total - passed
    
    print(f"\n总测试: {total}")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    
    if failed == 0:
        print("\n✓ 所有测试通过！可以开始运行贝叶斯优化")
        print("\n下一步:")
        print("  ./run_bayesian_tuning.sh")
        print("  或")
        print("  python bayesian_tuning_wrapper.py --n_trials 50")
    else:
        print("\n✗ 部分测试失败，请解决上述问题后再运行优化")
        print("\n失败的测试:")
        for name, passed in results.items():
            if not passed:
                print(f"  - {name}")
    
    print("="*80)


def main():
    print("\n" + "="*80)
    print("贝叶斯优化环境测试")
    print("="*80)
    print()
    
    results = {}
    
    # 运行所有测试
    results['依赖检查'] = test_dependencies()
    results['文件完整性'] = test_files()
    results['配置加载'] = test_config()
    results['数据路径'] = test_data_paths()
    results['GPU检查'] = test_gpu()
    results['模型导入'] = test_model_import()
    
    # 打印摘要
    print_summary(results)


if __name__ == "__main__":
    # 确保在项目根目录
    if not os.path.exists('config/GF5-1105.py'):
        print("错误: 请在项目根目录运行此脚本")
        print("cd /home/wlwang/sq/CDFS-D2AFC-2026-main")
        sys.exit(1)
    
    main()
