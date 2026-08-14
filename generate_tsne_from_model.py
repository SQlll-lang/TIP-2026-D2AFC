"""
独立的t-SNE可视化生成脚本
使用已保存的模型权重，无需重新训练即可生成t-SNE可视化图

使用方法:
    python generate_tsne_from_model.py --model_path model_weights/GF5-1105_dataset1_5shot.pth --config config/GF5-1105.py
"""

import numpy as np
import os
import argparse
import pickle
import importlib.util
import torch
from torch.autograd import Variable
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

from model.mapping import Mapping
from model.GLAI_Former import GLAIFormer
from utils.dataloader import get_target_dataset
from utils import utils


def load_model_and_generate_tsne(model_path, config_path):
    """
    加载模型权重并生成t-SNE可视化
    
    Args:
        model_path: 模型权重文件路径
        config_path: 配置文件路径
    """
    
    # 加载配置
    _spec = importlib.util.spec_from_file_location("exp_config", config_path)
    _config_mod = importlib.util.module_from_spec(_spec)
    assert _spec.loader is not None
    _spec.loader.exec_module(_config_mod)
    config = _config_mod.config
    train_opt = config['train_config']
    data_path = config['data_path']
    target_data = config['target_data']
    target_data_gt = config['target_data_gt']
    
    patch_size = train_opt['patch_size']
    TAR_INPUT_DIMENSION = train_opt['tar_input_dim']
    N_DIMENSION = train_opt['n_dim']
    emb_size = train_opt['d_emb']
    TAR_CLASS_NUM = train_opt['tar_class_num']
    TAR_LSAMPLE_NUM_PER_CLASS = train_opt['tar_lsample_num_per_class']
    GPU = config['gpu']
    
    print(f"Loading model from: {model_path}")
    print(f"Configuration: {TAR_CLASS_NUM} classes, {TAR_LSAMPLE_NUM_PER_CLASS}-shot learning")
    
    # 加载模型权重
    device = torch.device(f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    # 初始化模型
    mapping_tar = Mapping(TAR_INPUT_DIMENSION, N_DIMENSION).to(GPU)
    encoder = GLAIFormer(inp_channels=N_DIMENSION, dim=emb_size, depths=[1], 
                  num_heads_spa=[8], num_heads_spe=[7], dropout=0.3).to(GPU)
    
    # 加载权重
    mapping_tar.load_state_dict(checkpoint['mapping_tar'])
    encoder.load_state_dict(checkpoint['encoder'])
    
    # 设置为评估模式
    mapping_tar.eval()
    encoder.eval()
    
    print(f"Model loaded successfully! Best accuracy: {checkpoint['accuracy']:.2f}% at episode {checkpoint['episode']}")
    
    # 加载目标域数据（根据 target_data 自动选择数据集 loader）
    # 兼容 config 中 target_data / target_data_gt 既可能是“相对 data_path 的相对路径”，也可能是“绝对路径”
    def _resolve_path(base_dir: str, p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(base_dir, p)

    test_data = _resolve_path(data_path, target_data)
    test_label = _resolve_path(data_path, target_data_gt)

    lower_path = test_data.lower()
    if "gf51105" in lower_path or "gf5-1105" in lower_path:
        Data_Band_Scaler, GroundTruth = utils.load_data_custom_GF51105(test_data, test_label)
        print("Dataset detected: GF5-1105 (using load_data_custom_GF51105)")
    elif "gf50525" in lower_path or "gf5_0525" in lower_path or "0525" in lower_path or "gf5-0525" in lower_path:
        # GF5-0525 在本工程中使用 load_data_GF50525
        Data_Band_Scaler, GroundTruth = utils.load_data_GF50525(test_data, test_label)
        print("Dataset detected: GF5-0525 / GF50525 (using load_data_GF50525)")
    elif "ip" in lower_path or "indian" in lower_path or "indian_pines" in lower_path:
        Data_Band_Scaler, GroundTruth = utils.load_data_IP(test_data, test_label)
        print("Dataset detected: IP / Indian Pines (using load_data_IP)")
    elif "zy1" in lower_path or "zy10424" in lower_path:
        Data_Band_Scaler, GroundTruth = utils.load_data_custom_ZY10424(test_data, test_label)
        print("Dataset detected: ZY1-0424 (using load_data_custom_ZY10424)")
    else:
        raise ValueError(
            f"Unsupported dataset for t-SNE script.\n"
            f"test_data path: {test_data}\n"
            f"Currently supported: GF5-1105, GF5-0525(GF50525), IP(Indian Pines), ZY1-0424."
        )
    
    # 获取数据加载器
    train_loader, test_loader, _, _, G, RandPerm, Row, Column, nTrain, _, _ = get_target_dataset(
        Data_Band_Scaler=Data_Band_Scaler,
        GroundTruth=GroundTruth,
        class_num=TAR_CLASS_NUM,
        tar_lsample_num_per_class=TAR_LSAMPLE_NUM_PER_CLASS,
        shot_num_per_class=TAR_LSAMPLE_NUM_PER_CLASS,
        patch_size=patch_size
    )
    
    print("Extracting features...")
    with torch.no_grad():
        # 提取训练集特征
        train_datas, train_labels = train_loader.__iter__().__next__()
        train_features_tsne, _ = encoder(mapping_tar(Variable(train_datas).to(GPU)))
        train_features_tsne = train_features_tsne.cpu().numpy()
        train_labels_np = train_labels.numpy()
        
        # 提取测试集特征
        test_features_list = []
        test_labels_list = []
        for test_datas, test_labels in test_loader:
            test_features, _ = encoder(mapping_tar(Variable(test_datas).to(GPU)))
            test_features_list.append(test_features.cpu().numpy())
            test_labels_list.append(test_labels.numpy())
        
        test_features_tsne = np.concatenate(test_features_list, axis=0)
        test_labels_np = np.concatenate(test_labels_list, axis=0)
        
        # 合并特征
        all_features = np.concatenate([train_features_tsne, test_features_tsne], axis=0)
        all_labels = np.concatenate([train_labels_np, test_labels_np], axis=0)
    
    print(f"Total samples: {len(all_labels)} (Train: {len(train_labels_np)}, Test: {len(test_labels_np)})")
    print("Running t-SNE dimensionality reduction (this may take a few minutes)...")
    
    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_2d = tsne.fit_transform(all_features)
    
    # 分离训练和测试的2D坐标
    train_2d = features_2d[:len(train_labels_np)]
    test_2d = features_2d[len(train_labels_np):]
    
    print("Generating t-SNE visualization...")
    
    # 全局字体设置：坐标轴数字和文字加粗、清晰
    plt.rcParams['font.family'] = 'DejaVu Sans'  # Linux 常见字体
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    
    # 绘制t-SNE图
    plt.figure(figsize=(14, 11))
    colors = plt.cm.tab20(np.linspace(0, 1, TAR_CLASS_NUM))
    
    # 绘制测试集（用圆圈），类别标签从 1 开始显示
    for i in range(TAR_CLASS_NUM):
        mask = test_labels_np == i
        plt.scatter(test_2d[mask, 0], test_2d[mask, 1], 
                   c=[colors[i]], label=f'Class {i+1}', 
                   alpha=0.7, s=60, marker='o', edgecolors='none')
    
    # 并排展示时需更大字号，便于看清（再大一号）
    plt.title(f't-SNE Visualization (Accuracy: {checkpoint["accuracy"]:.2f}%)', 
             fontsize=30, fontweight='bold', pad=24)
    plt.xlabel('t-SNE Dimension 1', fontsize=26, fontweight='bold')
    plt.ylabel('t-SNE Dimension 2', fontsize=26, fontweight='bold')
    legend = plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=20,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.9,
    )
    # 单独设置图例边框样式，避免不支持的 linewidth 关键字参数
    try:
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(2)
    except Exception:
        pass
    plt.grid(True, alpha=0.4, linewidth=1.2)
    plt.tick_params(axis='both', which='major', labelsize=22, width=2, length=6)
    
    # 加粗坐标轴边框和刻度数字
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight('bold')
        tick.set_fontsize(22)
    
    plt.tight_layout()
    
    # 保存图片到 visualization_new 文件夹
    model_filename = os.path.basename(model_path).replace('.pth', '')
    tsne_save_path = f"visualization_new/tsne_{model_filename}_generated.png"
    os.makedirs(os.path.dirname(tsne_save_path), exist_ok=True)
    plt.savefig(tsne_save_path, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ t-SNE visualization saved to: {tsne_save_path}")
    
    # ==================== 误差分布热力图（子图1） ====================
    print("Generating error distribution heatmap...")
    with torch.no_grad():
        # 1. 使用同一批训练数据提取特征（与主程序保持一致）
        train_datas, train_labels = train_loader.__iter__().__next__()
        train_features, _ = encoder(mapping_tar(Variable(train_datas).to(GPU)))

        # 归一化到 [0, 1]
        max_value = train_features.max()
        min_value = train_features.min()
        train_features = (train_features - min_value) * 1.0 / (max_value - min_value)

        # 2. 基于训练特征构建 1-NN 分类器
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(train_features.cpu().detach().numpy(), train_labels.numpy())

        # 3. 对测试集进行预测
        predict_all = np.array([], dtype=np.int64)
        labels_all = np.array([], dtype=np.int64)

        for test_datas, test_labels in test_loader:
            test_features, _ = encoder(mapping_tar(Variable(test_datas).to(GPU)))
            test_features = (test_features - min_value) * 1.0 / (max_value - min_value)

            predict_labels = knn.predict(test_features.cpu().detach().numpy())
            test_labels_np = test_labels.numpy()

            predict_all = np.append(predict_all, predict_labels)
            labels_all = np.append(labels_all, test_labels_np)

        # 4. 构建误差矩阵（与原始 GroundTruth 尺寸一致）
        error_map = np.zeros((G.shape[0], G.shape[1]))

        # 利用 RandPerm / Row / Column / nTrain 将样本映射回空间位置
        for i in range(len(predict_all)):
            row_idx = Row[RandPerm[nTrain + i]]
            col_idx = Column[RandPerm[nTrain + i]]
            if predict_all[i] != labels_all[i]:
                error_map[row_idx, col_idx] = 1
            else:
                error_map[row_idx, col_idx] = 0

        # 去掉边缘 padding
        halfwidth = patch_size // 2
        error_map_crop = error_map[halfwidth:-halfwidth, halfwidth:-halfwidth]

        # 5. 绘制误差分布热力图（对应你说的“子图1：误差分布热力图”，单独成图）
        plt.figure(figsize=(10, 8))
        im = plt.imshow(
            error_map_crop,
            cmap='RdYlGn_r',
            interpolation='nearest',
            vmin=0,
            vmax=1
        )
        # 并排展示时需更大字号，便于看清（再大一号）
        plt.title('Error Distribution Map (Red: Wrong, Green: Correct)',
                  fontsize=28, fontweight='bold', pad=18)

        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label('Error (1=Wrong, 0=Correct)', fontsize=24, fontweight='bold')
        cbar.ax.tick_params(labelsize=20, width=2, length=6)
        cbar.outline.set_linewidth(2)

        # 误差图本身不需要坐标轴数字，关闭坐标轴
        plt.axis('off')

        plt.tight_layout()

        heatmap_save_path = f"visualization_new/error_heatmap_{model_filename}_generated.png"
        os.makedirs(os.path.dirname(heatmap_save_path), exist_ok=True)
        plt.savefig(
            heatmap_save_path,
            dpi=600,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
        plt.close()

    print(f"✓ Error heatmap saved to: {heatmap_save_path}")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate t-SNE visualization from saved model weights")
    parser.add_argument('--model_path', type=str, 
                       default='model_weights/GF5-1105_dataset1_5shot.pth',
                       help='Path to the saved model weights')
    parser.add_argument('--config', type=str, 
                       default='config/GF5-1105.py',
                       help='Path to the config file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        print("\nAvailable models:")
        if os.path.exists('model_weights'):
            for f in os.listdir('model_weights'):
                if f.endswith('.pth'):
                    print(f"  - model_weights/{f}")
        else:
            print("  No models found. Please train first!")
        exit(1)
    
    load_model_and_generate_tsne(args.model_path, args.config)

