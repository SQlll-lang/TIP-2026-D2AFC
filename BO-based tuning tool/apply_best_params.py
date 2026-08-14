"""
应用贝叶斯优化找到的最佳超参数到训练脚本
"""

import json
import glob
import os
import argparse
from datetime import datetime


def find_latest_results():
    """查找最新的优化结果文件"""
    result_files = glob.glob("GF51105_bayesian_opt_*_results.json")
    if not result_files:
        return None
    
    # 按修改时间排序，返回最新的
    result_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return result_files[0]


def load_best_params(result_file):
    """从JSON文件加载最佳参数"""
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    return results


def update_training_script(params, target_file='automated_tuning.py', backup=True):
    """
    更新训练脚本中的超参数
    
    参数:
        params: 最佳参数字典
        target_file: 目标训练脚本
        backup: 是否备份原文件
    """
    
    if not os.path.exists(target_file):
        print(f"错误: 文件不存在 {target_file}")
        return False
    
    # 读取原文件
    with open(target_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 备份原文件
    if backup:
        backup_file = f"{target_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"已备份原文件到: {backup_file}")
    
    # 替换参数
    lines = content.split('\n')
    modified_lines = []
    modifications = []
    
    for i, line in enumerate(lines):
        new_line = line
        
        # 1. 替换 lambda_weight (第205行附近)
        if 'ConTeXLoss' in line and 'lambda_weight=' in line:
            old_line = line
            # 提取当前值
            import re
            match = re.search(r'lambda_weight=([0-9.]+)', line)
            if match:
                old_value = match.group(1)
                new_value = params['best_params']['lambda_weight']
                new_line = line.replace(f'lambda_weight={old_value}', 
                                       f'lambda_weight={new_value}')
                if old_line != new_line:
                    modifications.append(f"第{i+1}行: lambda_weight: {old_value} -> {new_value}")
        
        # 2. 替换 mask_ratio (第424、426行附近)
        if 'random_mask_batch_spatial' in line and 'target_ssl_data' in line:
            old_line = line
            # 提取当前值
            import re
            match = re.search(r'random_mask_batch_spatial\([^,]+,\s*([0-9.]+)\)', line)
            if match:
                old_value = match.group(1)
                new_value = params['best_params']['mask_ratio']
                new_line = re.sub(
                    r'(random_mask_batch_spatial\([^,]+,\s*)([0-9.]+)(\))',
                    f'\\g<1>{new_value}\\g<3>',
                    line
                )
                if old_line != new_line:
                    modifications.append(f"第{i+1}行: mask_ratio: {old_value} -> {new_value}")
        
        # 3. 替换损失权重 (第523行附近)
        if 'loss = f_loss +' in line and 'CTx_loss_tar' in line and 'loss_wd' in line:
            old_line = line
            # 提取当前值
            import re
            # 匹配 f_loss + X * CTx_loss_tar + Y * loss_wd + Z * loss_disc
            match = re.search(
                r'loss\s*=\s*f_loss\s*\+\s*([0-9.]+)\s*\*\s*CTx_loss_tar\s*\+\s*([0-9.]+)\s*\*\s*loss_wd\s*\+\s*([0-9.]+)\s*\*\s*loss_disc',
                line
            )
            if match:
                old_ctx = match.group(1)
                old_wd = match.group(2)
                old_disc = match.group(3)
                
                new_ctx = params['best_params']['weight_ctx']
                new_wd = params['best_params']['weight_wd']
                new_disc = params['best_params']['weight_disc']
                
                new_line = f"        loss = f_loss + {new_ctx} * CTx_loss_tar + {new_wd} * loss_wd + {new_disc} * loss_disc"
                
                if old_line != new_line:
                    modifications.append(f"第{i+1}行: weight_ctx: {old_ctx} -> {new_ctx}")
                    modifications.append(f"第{i+1}行: weight_wd: {old_wd} -> {new_wd}")
                    modifications.append(f"第{i+1}行: weight_disc: {old_disc} -> {new_disc}")
        
        modified_lines.append(new_line)
    
    # 写入修改后的内容
    new_content = '\n'.join(modified_lines)
    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    return modifications


def generate_report(results, modifications):
    """生成修改报告"""
    
    print("\n" + "="*80)
    print("贝叶斯优化结果报告")
    print("="*80)
    
    print(f"\n优化信息:")
    print(f"  Study名称: {results['study_name']}")
    print(f"  完成时间: {results['timestamp']}")
    print(f"  试验次数: {results['n_trials']}")
    
    print(f"\n最佳结果:")
    print(f"  Trial编号: {results['best_trial']}")
    print(f"  准确率: {results['best_accuracy']:.2f}%")
    
    print(f"\n最佳超参数:")
    params = results['best_params']
    print(f"  lambda_weight:  {params['lambda_weight']:.4f}")
    print(f"  mask_ratio:     {params['mask_ratio']:.4f}")
    print(f"  weight_ctx:     {params['weight_ctx']:.4f}")
    print(f"  weight_wd:      {params['weight_wd']:.6f}")
    print(f"  weight_disc:    {params['weight_disc']:.4f}")
    
    if modifications:
        print(f"\n应用的修改:")
        for mod in modifications:
            print(f"  ✓ {mod}")
    else:
        print(f"\n未应用任何修改（参数可能已经是最优值）")
    
    print("\n" + "="*80)
    print("下一步:")
    print("  1. 检查修改后的 automated_tuning.py 文件")
    print("  2. 运行完整训练: python automated_tuning.py")
    print("  3. 如需回滚，使用备份文件")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="应用贝叶斯优化的最佳超参数")
    parser.add_argument('--result_file', type=str, default=None,
                       help='结果JSON文件路径（默认: 自动查找最新）')
    parser.add_argument('--target_file', type=str, default='automated_tuning.py',
                       help='目标训练脚本路径（默认: automated_tuning.py）')
    parser.add_argument('--no_backup', action='store_true',
                       help='不备份原文件')
    parser.add_argument('--dry_run', action='store_true',
                       help='仅显示将要进行的修改，不实际修改文件')
    args = parser.parse_args()
    
    # 查找结果文件
    if args.result_file is None:
        print("查找最新的优化结果...")
        result_file = find_latest_results()
        if result_file is None:
            print("错误: 未找到优化结果文件")
            print("请先运行贝叶斯优化: python bayesian_tuning_wrapper.py")
            return
        print(f"找到结果文件: {result_file}")
    else:
        result_file = args.result_file
        if not os.path.exists(result_file):
            print(f"错误: 结果文件不存在: {result_file}")
            return
    
    # 加载最佳参数
    print("加载最佳参数...")
    results = load_best_params(result_file)
    
    # 显示结果
    print("\n" + "="*80)
    print(f"最佳准确率: {results['best_accuracy']:.2f}%")
    print(f"最佳超参数:")
    for key, value in results['best_params'].items():
        print(f"  {key}: {value}")
    print("="*80)
    
    if args.dry_run:
        print("\n[Dry Run模式] 不会修改文件")
        return
    
    # 询问用户确认
    response = input("\n是否应用这些参数到训练脚本? (y/n): ")
    if response.lower() != 'y':
        print("已取消")
        return
    
    # 更新训练脚本
    print(f"\n正在更新 {args.target_file}...")
    modifications = update_training_script(
        results,
        target_file=args.target_file,
        backup=not args.no_backup
    )
    
    if modifications:
        print(f"✓ 成功应用 {len(modifications)} 处修改")
    else:
        print("未进行任何修改")
    
    # 生成报告
    generate_report(results, modifications)


if __name__ == "__main__":
    main()
