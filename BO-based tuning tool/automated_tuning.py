import numpy as np
import os
import argparse
import pickle
import time
import importlib.util
import random
import copy
# import importlib.util
import logging
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

# 贝叶斯优化库（训练本身不依赖；调参脚本才需要）
try:
    import optuna  # noqa: F401
    from optuna.trial import TrialState  # noqa: F401
except Exception:
    optuna = None
    TrialState = None
import json

from model.mapping import Mapping
from utils.dataloader import get_HBKC_data_loader, Task, get_target_dataset, tagetSSLDataset
from utils import utils, data_augment
from model.GLAI_Former import GLAIFormer, CrossTransformer, DomainDiscriminator

from model import loss
from model.loss import ConTeXLoss

# t-SNE可视化分析
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# import warnings

# warnings.simplefilter(action='ignore', category=FutureWarning)


parser = argparse.ArgumentParser(description="Few Shot Visual Recognition with Bayesian Optimization")
parser.add_argument('--config', type=str, default=os.path.join('./config', 'GF5-1105.py'))
parser.add_argument('--n_trials', type=int, default=50, help='贝叶斯优化的试验次数')
parser.add_argument('--timeout', type=int, default=None, help='优化的最大时间（秒）')
parser.add_argument('--run_once', action='store_true', help='单次运行（用于贝叶斯优化脚本调用）')
parser.add_argument('--lambda_weight', type=float, default=0.0, help='ConTeXLoss 的 lambda_weight')
parser.add_argument('--mask_ratio', type=float, default=0.8, help='random_mask_batch_spatial 的 mask_ratio')
parser.add_argument('--weight_ctx', type=float, default=2.0, help='CTx_loss_tar 的权重')
parser.add_argument('--weight_wd', type=float, default=0.001, help='loss_wd 的权重')
parser.add_argument('--weight_disc', type=float, default=1.0, help='loss_disc 的权重')
parser.add_argument('--max_episode', type=int, default=None, help='覆盖配置中的 episode（加速调参）')
parser.add_argument('--seeds', type=str, default=None, help='覆盖 seeds（逗号分隔，如 1236,1237）')
args = parser.parse_args()

# load hyperparameters
_spec = importlib.util.spec_from_file_location("exp_config", args.config)
_config_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_config_mod)
config = _config_mod.config
# 全局配置
train_opt = config['train_config']
data_path = config['data_path']
source_data = config['source_data']
target_data = config['target_data']
target_data_gt = config['target_data_gt']
log_dir = config['log_dir']
patch_size = train_opt['patch_size']
batch_task = train_opt['batch_task']
emb_size = train_opt['d_emb']
SRC_INPUT_DIMENSION = train_opt['src_input_dim']
TAR_INPUT_DIMENSION = train_opt['tar_input_dim']
N_DIMENSION = train_opt['n_dim']
SHOT_NUM_PER_CLASS = train_opt['shot_num_per_class']
QUERY_NUM_PER_CLASS = train_opt['query_num_per_class']
EPISODE = train_opt['episode']
if args.max_episode is not None:
    EPISODE = min(EPISODE, int(args.max_episode))
LEARNING_RATE = train_opt['lr']
GPU = config['gpu']
TAR_CLASS_NUM = train_opt['tar_class_num']
TAR_LSAMPLE_NUM_PER_CLASS = train_opt['tar_lsample_num_per_class']
SCR_CLASS_NUM = train_opt['scr_class_num']
WEIGHT_DECAY = train_opt['weight_decay']

utils.same_seeds(0)

# get src/tar class number -> label semantic vector
labels_src = ["Breeding Area", "Salt Pan Crystallization Pond", "Field", "Reed", "Salt Field",
              "Natural Willow Forest", "Other Forest Land", "Reed and Tamarisk Mixed Vegetation", "Tamarisk and Saltbush Mixed Vegetation", "Saline Land Saltbush Growth Area",
              "Pit Pond", "Oil Well Platform", "Saline-Alkaline Flats", "Bare Beach", "Yellow River", "Reed and Saltbush Mixed Vegetation",
              "Reed, Saltbush, and Tamarisk Mixed Vegetation", "Sea", "Smooth Cordgrass Management Area"]

labels_src_new = [
    # 前 14 项：labels_tar 原封不动
    "Sea", "Yellow River", "Oil Well Platform", "Tamarisk and Saltbush Mixed Vegetation", "Saline‑Alkaline Flats", "Bare Beach", "Pit Pond",
    "Reed and Saltbush Mixed Vegetation", "Reed, Saltbush, and Tamarisk Mixed Vegetation", "Reed", "Reed and Tamarisk Mixed Vegetation",
    "Natural Willow Forest", "Saline Land Saltbush Growth Area", "Smooth Cordgrass Management Area",
    # 后面 5 项：源域中不在 labels_tar 的类别，保持原来顺序
    "Breeding Area", "Salt Pan Crystallization Pond", "Field", "Salt Field","Other Forest Land",
]

# GF5-1105
labels_tar = ["Sea", "Yellow River", "Oil Well Platform", "Tamarisk and Saltbush Mixed Vegetation", "Saline-Alkaline Flats", "Bare Beach", "Pit Pond",
              "Reed and Saltbush Mixed Vegetation", "Reed, Saltbush, and Tamarisk Mixed Vegetation", "Reed", "Reed and Tamarisk Mixed Vegetation", "Natural Willow Forest",
              "Saline Land Saltbush Growth Area", "Smooth Cordgrass Management Area"]

# 构建类别映射  使得相同类别的标签相同
# 1) 构造新的源域标签顺序（名称列表）：
#    先把目标域的 C 个共有类放前面，再把源域中剩下的（不在labels_tar里）按原顺序追加
new_labels_src = labels_tar + [lbl for lbl in labels_src if lbl not in set(labels_tar)]

# 2) 构造“旧索引 → 新索引”映射（针对源域 labels_train）
src_old2new = {
    old_idx: new_labels_src.index(lbl)
    for old_idx, lbl in enumerate(labels_src)
}
print("新的源域标签顺序 (new_labels_src):")
for idx, lbl in enumerate(new_labels_src):
    print(f"{idx:2d}: {lbl}")

# from transformers import BertModel, BertTokenizer
#
# # 标签特征提取器
# model = BertModel.from_pretrained('pretrain-model/bert-base-uncased')    # 加载预训练模型
# model.eval()              # 切换成评估模式
# tokenizer = BertTokenizer.from_pretrained('pretrain-model/bert-base-uncased')   # 加载与模型配套的分词器，确保分词方式与模型预训练时一致
#
# encoded_inputs_src = tokenizer(labels_src, padding=True, truncation=True, return_tensors='pt')  # 对源域标签进行分词  注意使用的是映射前的源域还是映射后的
# with torch.no_grad():
#     outputs_src = model(**encoded_inputs_src)     # 获取模型输出
#
#  # 提取每个隐藏层输出的第一个 token（[CLS]）的向量，形状为 (num_classes, 768)，用于表示类别的语义向量   # [CLS]向量通常用于表示整个序列的语义
# semantic_mapping_src = outputs_src.last_hidden_state[:, 0, :]  # (num_classess, 768)
#
# encoded_inputs_tar = tokenizer(labels_tar, padding=True, truncation=True, return_tensors='pt')
# with torch.no_grad():
#     outputs_tar = model(**encoded_inputs_tar)
# semantic_mapping_tar = outputs_tar.last_hidden_state[:, 0, :]  # (num_classess, 768)
#
# semantic_mapping_src = semantic_mapping_src.cpu().numpy()    # 将pytorch张量转化为numpy数组
# semantic_mapping_tar = semantic_mapping_tar.cpu().numpy()

# load source domain data
print('----------------------------load source domain data----------------------------')
with open(os.path.join(data_path, source_data), 'rb') as handle:    # 加载源域数据
    source_imdb = pickle.load(handle)

data_train = source_imdb['data']
labels_train = source_imdb['Labels']
labels_train_aligned = [src_old2new[old] for old in labels_train]
labels_train = labels_train_aligned

for _ in range(5):
    i = random.randrange(len(labels_train))
    old_label = labels_train[i]
    new_label = labels_train_aligned[i]
    print(f"样本 {i}: 原始标签 {old_label} ({labels_src[old_label]}), 对齐后 {new_label} ({new_labels_src[new_label]})")

keys_all_train = sorted(list(set(labels_train)))     # 去重标签，对类别进行排序
label_encoder_train = {}                      # 创建一个字典，将每个类别映射为唯一的索引
for i in range(len(keys_all_train)):
    label_encoder_train[keys_all_train[i]] = i
train_dict = {}                                        # 构建训练字典
for class_, path in zip(labels_train, data_train):    # 将标签列表labels_train和数据路径列表data_train逐项配对，形成(class_, path)的元组
    if label_encoder_train[class_] not in train_dict:     # 将类别标签 class_ 转换为对应的类别索引
        train_dict[label_encoder_train[class_]] = []
    train_dict[label_encoder_train[class_]].append(path)
del keys_all_train
del label_encoder_train

metatrain_data = utils.sanity_check(train_dict)

# 遍历metatrain_data中的每个类别的数据，将其维度从(H, W, C)变为(C, H, W)
for class_ in metatrain_data:
    for i in range(len(metatrain_data[class_])):
        metatrain_data[class_][i] = np.transpose(metatrain_data[class_][i], (2, 0, 1))

# source domain adaptation data
print(source_imdb['data'].shape)
source_imdb['data'] = source_imdb['data'].transpose((1, 2, 3, 0))
print(source_imdb['data'].shape)
print(source_imdb['Labels'].shape)
source_dataset = utils.matcifar(source_imdb, train=True, d=3, medicinal=0)
source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=128, shuffle=True, num_workers=0, drop_last=True)
del source_dataset, source_imdb

# load target data
print('--------------------------------load target data-------------------------------')
test_data = os.path.join(data_path, target_data)
test_label = os.path.join(data_path, target_data_gt)
Data_Band_Scaler, GroundTruth = utils.load_data_custom_GF51105(test_data, test_label)  # 加载测试数据和标签

# loss init
crossEntropy = nn.CrossEntropyLoss().to(GPU)

wd_loss = loss.SinkhornDistance(eps=0.1, max_iter=100, reduction='mean')   # Wasserstein距离损失  用于全局分布域对齐
domain_criterion_2 = nn.BCEWithLogitsLoss().to(GPU)    # 二进制交叉熵损失 域判别损失 D2

contex_loss_fn = ConTeXLoss(temperature=0.1, lambda_weight=float(args.lambda_weight)).to(GPU)

# awd_da_loss = loss.AWD_DA_Loss(epsilon=0.05)  # 自适应沃瑟斯坦距离损失


# experimental result index
# 默认只跑前 2 个 seed；如传入 --seeds 则按传入 seed 数量运行
# seeds = [1236, 1237, 1226, 1227, 1211, 1212, 1216, 1240, 1222, 1223]
default_seeds = [1236, 1237, 1230, 1238, 1211, 1210, 1216, 1240, 1235, 1223]
if args.seeds is not None and str(args.seeds).strip() != "":
    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip() != ""]
else:
    seeds = default_seeds[:2]
nDataSet = len(seeds)
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, TAR_CLASS_NUM])   # 存储每次实验中每个类别的结果
k = np.zeros([nDataSet, 1])
f1_scores = np.zeros([nDataSet, 1])  # 存储F1-score
best_predict_all = []   # 用于存储所有实验中的最佳预测结果
best_predict_full = []  # 用于存储所有实验中的最佳全像素预测结果
best_full_G, best_Row_all, best_Column_all, best_nTrain_all = None, None, None, None
best_G, best_RandPerm, best_Row, best_Column, best_nTrain = None, None, None, None, None   # 存储实验过程中的最佳参数

# seeds 由上方逻辑决定

# log setting
experimentSetting = '{}way_{}shot_{}'.format(TAR_CLASS_NUM, TAR_LSAMPLE_NUM_PER_CLASS, target_data.split('/')[0])
utils.set_logging_config(os.path.join(log_dir, experimentSetting), nDataSet)   # 存储实验日志
logger = logging.getLogger('main')    # logging.getLogger('main')：获取名为 main 的日志记录器；该记录器用于输出日志信息，例如调试、错误、警告等
logger.info('seeds_list:{}'.format(seeds))    # 记录种子列表


for iDataSet in range(nDataSet):
    logger.info('emb_size:{}'.format(emb_size))
    logger.info('patch_size:{}'.format(patch_size))
    logger.info('seeds:{}'.format(seeds[iDataSet]))

    utils.same_seeds(seeds[iDataSet])    # 确保在训练过程中使用相同的随机种子，从而使得实验结果可复现

    # 加载包括目标域训练数据、测试数据、目标域元训练（增强之后）数据、填充后的图像大小、数据位置索引、行、列、训练样本数、增强数据平铺列表数据、增强数据标签平铺列表数据
    train_loader, test_loader, target_da_metatrain_data, target_loader, G, RandPerm, Row, Column, nTrain, target_aug_data_ssl, target_aug_label_ssl = get_target_dataset(
        Data_Band_Scaler=Data_Band_Scaler,
        GroundTruth=GroundTruth,
        class_num=TAR_CLASS_NUM,
        tar_lsample_num_per_class=TAR_LSAMPLE_NUM_PER_CLASS,
        shot_num_per_class=TAR_LSAMPLE_NUM_PER_CLASS,
        patch_size=patch_size)

    # 单独调用你新写的全像素 loader
    # all_test_loader, G_pad, Row_all, Col_all, nTrain_all = get_allpixel_loader(
    #     Data_Band_Scaler=Data_Band_Scaler,
    #     GroundTruth=GroundTruth,
    #     HalfWidth=patch_size // 2

    # )

    target_ssl_dataset = tagetSSLDataset(target_aug_data_ssl, target_aug_label_ssl)      # 目标域数据加载器
    target_ssl_dataloader = torch.utils.data.DataLoader(target_ssl_dataset, batch_size=64, shuffle=True, drop_last=True)

    # 生成小样本学习任务所需的掩码和参数  一个小样本任务的支持集大小、总样本大小、查询集掩码
    num_supports, num_samples, query_edge_mask, evaluation_mask = utils.preprocess(TAR_CLASS_NUM, SHOT_NUM_PER_CLASS,
                                                                                   QUERY_NUM_PER_CLASS, batch_task, GPU)
    # 定义源域和目标域的映射网络（将原始特征映射到公共空间）与特征提取网络Encoder
    mapping_src = Mapping(SRC_INPUT_DIMENSION, N_DIMENSION).to(GPU)
    mapping_tar = Mapping(TAR_INPUT_DIMENSION, N_DIMENSION).to(GPU)
    # domain_classifier = DomainClassifier().to(GPU)
    # random_layer = RandomLayer([emb_size, TAR_CLASS_NUM], 1024).to(GPU)
    cross_transformer = CrossTransformer(embed_dim=emb_size, num_heads=8) # 初始化 Cross-Transformer
    domain_classifier_2 = DomainDiscriminator().to(GPU)
    # lfcn = LFCN(input_dim=emb_size)   # 计算自适应代价矩阵的轻量化网络


    encoder = GLAIFormer(inp_channels=N_DIMENSION, dim=emb_size, depths=[1], num_heads_spa=[8], num_heads_spe=[7], dropout=0.3).to(GPU)
    # encoder = Net_xiaorong(inp_channels=N_DIMENSION, dim=emb_size, depths=[1], num_heads_spa=[8], num_heads_spe=[7], dropout=0.3).to(GPU)



    mapping_src_optim = torch.optim.SGD(mapping_src.parameters(), lr=LEARNING_RATE, momentum=0.9,
                                        weight_decay=WEIGHT_DECAY)
    mapping_tar_optim = torch.optim.SGD(mapping_tar.parameters(), lr=LEARNING_RATE, momentum=0.9,
                                        weight_decay=WEIGHT_DECAY)
    encoder_optim = torch.optim.SGD(encoder.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)

    # domain_classifier_optim = torch.optim.SGD(domain_classifier.parameters(), lr=LEARNING_RATE, momentum=0.9,
    #                                           weight_decay=WEIGHT_DECAY)

    cross_transformer_optim = torch.optim.SGD(cross_transformer.parameters(), lr=LEARNING_RATE, momentum=0.9,
                                              weight_decay=WEIGHT_DECAY)
    domain_classifier_2_optim = torch.optim.SGD(domain_classifier_2.parameters(), lr=LEARNING_RATE, momentum=0.9,
                                              weight_decay=WEIGHT_DECAY)
    # lfcn_optim = torch.optim.SGD(domain_classifier_2.parameters(), lr=LEARNING_RATE, momentum=0.9,
    #                                           weight_decay=WEIGHT_DECAY)


    # 应用自定义权重初始化
    mapping_src.apply(utils.weights_init)
    mapping_tar.apply(utils.weights_init)
    # domain_classifier.apply(utils.weights_init)
    domain_classifier_2.apply(utils.weights_init)
    # encoder.apply(utils.weights_init)
    # lfcn.apply(utils.weights_init)

    mapping_src.to(GPU)
    mapping_tar.to(GPU)
    # domain_classifier.to(GPU)
    domain_classifier_2.to(GPU)
    cross_transformer.to(GPU)
    encoder.to(GPU)
    # lfcn.to(GPU)

    # 设置为训练模式（启用Dropout等训练专用层）
    mapping_src.train()
    mapping_tar.train()
    # domain_classifier.train()
    domain_classifier_2.train()
    encoder.train()
    cross_transformer.train()
    # lfcn.train()

    logger.info("Training...")
    last_accuracy = 0.0                # 记录上一次准确率
    best_episode = 0                   # 最佳训练轮次

    # 保存“当前最优”模型参数快照（避免未命中更新分支导致未定义；同时用 deepcopy 防止后续训练覆盖）
    best_mapping_src_state = copy.deepcopy(mapping_src.state_dict())
    best_mapping_tar_state = copy.deepcopy(mapping_tar.state_dict())
    best_encoder_state = copy.deepcopy(encoder.state_dict())
    best_cross_transformer_state = copy.deepcopy(cross_transformer.state_dict())
    # 保存“训练前（初始化）”模型参数快照：用于三阶段对比（训练前 / 域对齐前 / 域对齐后）
    init_mapping_src_state = copy.deepcopy(mapping_src.state_dict())
    init_mapping_tar_state = copy.deepcopy(mapping_tar.state_dict())
    init_encoder_state = copy.deepcopy(encoder.state_dict())
    init_cross_transformer_state = copy.deepcopy(cross_transformer.state_dict())
    # 源域正确预测数/总样本数  目标域相同统计  当前准确率
    total_hit_src, total_num_src, total_hit_tar, total_num_tar, acc_src, acc_tar = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    train_start = time.time()
    writer = SummaryWriter()       # 创建可视化日志写入器

    # 获取目标域数据迭代器（用于masked监督对比学习）
    target_ssl_iter = iter(target_ssl_dataloader)       # 可无限迭代的数据加载器

    # 获取域对齐数据迭代器
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)


    for episode in range(EPISODE):                # 遍历所有训练轮次（元学习中的episode概念）
        # 源域任务构建
        task_src = Task(metatrain_data, TAR_CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)
        # 支持集数据加载器构建
        support_dataloader_src = get_HBKC_data_loader(task_src, num_per_class=SHOT_NUM_PER_CLASS, split="train",
                                                      shuffle=False)
        # 查询集数据加载器构建
        query_dataloader_src = get_HBKC_data_loader(task_src, num_per_class=QUERY_NUM_PER_CLASS, split="test",
                                                    shuffle=False)

        # 测试阶段只在目标域的 TAR_CLASS_NUM 个类别上进行，因此训练阶段必须模拟相同类别数的任务结构
        # 目标域任务构建
        task_tar = Task(target_da_metatrain_data, TAR_CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)
        support_dataloader_tar = get_HBKC_data_loader(task_tar, num_per_class=SHOT_NUM_PER_CLASS, split="train",
                                                      shuffle=False)
        query_dataloader_tar = get_HBKC_data_loader(task_tar, num_per_class=QUERY_NUM_PER_CLASS, split="test",
                                                    shuffle=False)

        # 加载源域数据
        support_src, support_label_src = support_dataloader_src.__iter__().__next__()
        query_src, query_label_src = query_dataloader_src.__iter__().__next__()

        support_real_labels_src = task_src.support_real_labels
        support_real_labels_tar = task_tar.support_real_labels

        # 目标域同理
        support_tar, support_label_tar = support_dataloader_tar.__iter__().__next__()
        query_tar, query_label_tar = query_dataloader_tar.__iter__().__next__()

        # print('support_src', support_src.shape)   # [14, 76, 9, 9]
        # print('query_src', query_src.shape)       # [266, 76, 9, 9]
        # print('support_tar', support_tar.shape)   # [14, 150, 7, 7]
        # print('query_tar', query_tar.shape)       # [266, 150, 7, 7]

        # 源域特征提取
        support_features_src, support_output_src = encoder(mapping_src(support_src.to(GPU)))  # (14, 128)
        query_features_src, query_output_src = encoder(mapping_src(query_src.to(GPU)))            # (266, 128)

        # 目标域特征提取（同理）
        support_features_tar, support_output_tar = encoder(mapping_tar(support_tar.to(GPU)))  # (14, 128)    包含一个语义特征的映射过程
        query_features_tar, query_output_tar = encoder(mapping_tar(query_tar.to(GPU)))   # torch.Size([266, 128])

        # 自适应沃瑟斯坦距离域对齐
        # src_feature = torch.cat([support_features_src, query_features_src], dim=0)
        # tar_feature = torch.cat([support_features_tar, query_features_tar], dim=0)
        # C_A = lfcn(src_feature, tar_feature)
        # loss_wd, _ = awd_da_loss(C_A)

        # 原型计算（Prototypical Networks核心）
        if SHOT_NUM_PER_CLASS > 1:          # 多样本时取平均
            support_proto_src = support_features_src.reshape(TAR_CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)
            support_proto_tar = support_features_tar.reshape(TAR_CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)

        else:                               # 单样本直接使用
            support_proto_src = support_features_src
            support_proto_tar = support_features_tar

        # 分类损失（交叉熵）
        logits_src = utils.euclidean_metric(query_features_src, support_proto_src)     # 欧氏距离计算相似度
        f_loss_src = crossEntropy(logits_src, query_label_src.long().to(GPU))          # 源域分类损失

        logits_tar = utils.euclidean_metric(query_features_tar, support_proto_tar)
        f_loss_tar = crossEntropy(logits_tar, query_label_tar.long().to(GPU))          # 目标域分类损失

        f_loss = f_loss_src + f_loss_tar  # 总分类损失


        # 目标域监督对比学习
        try:       # 获取增强数据
            target_ssl_data, target_ssl_label = next(target_ssl_iter)
        except Exception as err:       # 重新初始化迭代器
            target_ssl_iter = iter(target_ssl_dataloader)
            target_ssl_data, target_ssl_label = next(target_ssl_iter)

        # 数据增强（随机遮挡）
        augment1_target_ssl_data = torch.FloatTensor(
            data_augment.random_mask_batch_spatial(target_ssl_data.data.cpu(), float(args.mask_ratio)))  # (64, 150, 7, 7) 116 5 113 1 113 2
        augment2_target_ssl_data = torch.FloatTensor(
            data_augment.random_mask_batch_spatial(target_ssl_data.data.cpu(), float(args.mask_ratio)))  # (64, 150, 7, 7)
        augment_target_ssl_data = torch.cat((augment1_target_ssl_data, augment2_target_ssl_data),
                                            dim=0)  #````````````````````````````````````````` ([128, 150, 7, 7])
        features_augment, _ = encoder(mapping_tar(augment_target_ssl_data.to(GPU)))  # (128, 128)   特征提取

        augment1_target_ssl_feature = F.normalize(features_augment[:len(target_ssl_data), :], dim=1)  # (64, 128)   标准化
        augment2_target_ssl_feature = F.normalize(features_augment[len(target_ssl_data):, :], dim=1)  # (64, 128)
        augment_target_ssl_feature = torch.cat(
            [augment1_target_ssl_feature.unsqueeze(1), augment2_target_ssl_feature.unsqueeze(1)],
            dim=1)  # (128, 2, 128)
        # scl_loss_tar = SupConLoss_t(augment_target_ssl_feature, target_ssl_label, None, GPU)   # 对比损失计算
        # CTx_loss_tar = SupConLoss_t(augment_target_ssl_feature, target_ssl_label, None, GPU)  # 对比损失计算

        # Contex loss
        B, dim_ssl, _, _ = augment1_target_ssl_data.shape
        label_ssl = torch.cat([target_ssl_label, target_ssl_label], dim=0).to(GPU)  # (128, 128)
        feature_ssl = features_augment
        # 3. 生成 sample_ids
        ids = torch.arange(B, device=GPU)
        sample_ids = torch.cat([ids, ids], dim=0)  # [128]
        # 假设已实例化
        CTx_loss_tar = contex_loss_fn(feature_ssl, label_ssl, sample_ids)


        # get domain adaptation data from  source domain and target domain  获取域对齐数据（进行整体分布域对齐）
        try:
            source_data, source_label = next(source_iter)
        except Exception as err:
            source_iter = iter(source_loader)
            source_data, source_label = next(source_iter)
        try:
            target_data, target_label = next(target_iter)
        except Exception as err:
            target_iter = iter(target_loader)
            target_data, target_label = next(target_iter)
        # 源域和目标域特征掩码增强
        # augment1_src_data = torch.FloatTensor(
        #     data_augment.random_mask_batch_spatial(source_data.data.cpu(), 0.8))  # (64, 150, 7, 7)
        # augment1_tar_data = torch.FloatTensor(
        #     data_augment.random_mask_batch_spatial(target_data.data.cpu(), 0.8))  # (64, 150, 7, 7)
        # augment1_src_data = mapping_src(augment1_src_data.to(GPU))
        # augment1_tar_data = mapping_tar(augment1_tar_data.to(GPU))
        # augment_data = torch.cat((augment1_src_data, augment1_tar_data),
        #                                     dim=0)
        # # 生成多源聚合特征与对应标签
        # features_augment_src_tar, _ = encoder((augment_data.to(GPU)))  # (256, 128)   特征提取
        # augment_src_feature = F.normalize(features_augment_src_tar[:len(source_data), :], dim=1)  # (128, 128)   标准化
        # augment_tar_feature = F.normalize(features_augment_src_tar[len(target_data):, :], dim=1)  # (128, 128)
        # augment_tar_src_feature = torch.cat(
        #     [augment_src_feature.unsqueeze(1), augment_tar_feature.unsqueeze(1)],)
        # labels_src_tar = torch.cat((source_label, target_label),
        #                                     dim=0)  # ([256, ])
        # # 监督对比损失
        # scl_loss_tar = SupConLoss_t(augment_tar_src_feature, labels_src_tar, None, GPU)   # 对比损失计算

        # 特征提取
        wd_s, _ = encoder(mapping_src(source_data.to(GPU)))  # (64, 128)   特征提取
        wd_t, _ = encoder(mapping_tar(target_data.to(GPU)))  # (64, 128)   特征提取
        loss_wd, _, _ = wd_loss(wd_s, wd_t)                         # 后面尝试在特征后面用
        # loss_mmd = mmd_loss(wd_s, wd_t)
        # loss_coral = coral_loss(wd_s, wd_t)   # 计算 CORAL 损失，输入源域和目标域的特征矩阵

        # 利用域判别器进行域对齐
        # features = torch.cat([support_features_src, query_features_src, support_features_tar, query_features_tar], dim=0)
        # outputs = torch.cat([support_output_src, query_output_src, support_output_tar, query_output_tar], dim=0)
        # softmax_output = nn.Softmax(dim=1)(outputs)
        # labels = torch.cat([support_label_src, query_label_src, support_label_tar, query_label_tar], dim=0)
        # labels_hot = one_hot(labels, TAR_CLASS_NUM).to(GPU)
        # combined = torch.cat([features, labels_hot], dim=1)
        # # 给每个样本创建域标签
        # domain_label = torch.zeros([support_features_src.shape[0] + query_features_src.shape[0]
        #                             + support_features_tar.shape[0] + query_features_tar.shape[0], 1]).to(GPU)
        # domain_label[:support_features_src.shape[0] + query_features_src.shape[0]] = 1  # 将源域支持样本和查询样本的域标签设置为1，目标域样本设为0
        # randomlayer_out = random_layer.forward([features, softmax_output])
        #
        # domain_logits = domain_classifier(randomlayer_out, episode)
        # domain_loss = domain_criterion(domain_logits, domain_label)  # 计算域损失

        # 分布级域适应对齐
        features_src = torch.cat([support_features_src, query_features_src], dim=0)
        features_tar = torch.cat([support_features_tar, query_features_tar], dim=0)
        # 得到分布对齐后的特征 f_s_d, f_t_d (各 (280, 128))
        f_s_d, f_t_d = cross_transformer(features_src, features_tar)  # torch.Size([140, 128])
        # 将对齐后的特征拼接，作为判别器输入： (20, 512)
        F_d = torch.cat([f_s_d, f_t_d], dim=0)
        # 得到域分类 logits (20, 1)
        domain_logits_2 = domain_classifier_2(F_d).to(GPU)
        # 构造真实域标签：前 10 行为 1（源域），后 10 行为 0（目标域）
        domain_label_2 = torch.cat([torch.ones(f_s_d.size(0), 1), torch.zeros(f_t_d.size(0), 1)], dim=0).to(GPU)
        # 计算对抗损失（判别器方向）
        loss_disc = domain_criterion_2(domain_logits_2, domain_label_2)

        # 距离损失
        # loss_wd, _, _ = wd_loss(features_src, features_tar)


        # 总损失
        total_loss = f_loss + float(args.weight_ctx) * CTx_loss_tar + float(args.weight_wd) * loss_wd + float(args.weight_disc) * loss_disc
        # loss = f_loss + 2 * CTx_loss_tar


        # 消融实验
        # loss = f_loss  + 0.001 * loss_wd
        # loss = f_loss + 1 * loss_disc
        # loss = f_loss   # 116第1个 第2个为去掉LGIA



        # scl_loss_tar [0.5, 1, 1.5, 2, 2.5,3.0]
        # loss_wd [ 1, 0.1, 0.01, 0.001]  0.1、1、10、100都不行
        # loss_disc {0.01, 0.1, 1, 10}


        mapping_src.zero_grad()
        mapping_tar.zero_grad()
        encoder.zero_grad()
        # domain_classifier_optim.zero_grad()
        domain_classifier_2_optim.zero_grad()
        cross_transformer_optim.zero_grad()
        # lfcn_optim.zero_grad()

        total_loss.backward()   # 反向传播

        # 参数更新
        mapping_src_optim.step()
        mapping_tar_optim.step()
        encoder_optim.step()
        # domain_classifier_optim.step()   # 域判别器
        domain_classifier_2_optim.step()
        cross_transformer_optim.step()
        # lfcn_optim.step()

        # 准确率统计
        total_hit_src += torch.sum(torch.argmax(logits_src, dim=1).cpu() == query_label_src).item()
        total_num_src += query_src.shape[0]
        acc_src = total_hit_src / total_num_src    # 源域准确率

        # 目标域同理...
        total_hit_tar += torch.sum(torch.argmax(logits_tar, dim=1).cpu() == query_label_tar).item()
        total_num_tar += query_tar.shape[0]
        acc_tar = total_hit_tar / total_num_tar

        if (episode + 1) % 100 == 0:        # 定期日志记录
            logger.info(
                'episode: {:>3d}, f_loss: {:6.4f}, loss_wd: {:6.4f}, domain_loss_2: {:6.4f}, CTx_loss_tar: {:6.4f}, loss: {:6.4f}, acc_src: {:6.4f}, acc_tar: {:6.4f}'.format(
                    episode + 1,
                    f_loss.item(),
                    # loss_coral.item(),   loss_coral: {:6.4f},
                    loss_wd.item(),
                    # domain_loss.item(),domain_loss: {:6.4f},
                    loss_disc.item(),
                    # scl_loss_tar.item(), scl_loss_tar: {:6.4f},
                    CTx_loss_tar.item(),
                    # loss_lmmd.item(), loss_lmmd: {:6.4f}，
                    total_loss.item(),
                    acc_src,
                    acc_tar))

            writer.add_scalar('Loss/f_loss', f_loss.item(), episode + 1)

            # writer.add_scalar('Loss/loss_coral', loss_coral.item(), episode + 1)
            writer.add_scalar('Loss/loss_wd', loss_wd.item(), episode + 1)
            # writer.add_scalar('Loss/domain_loss', domain_loss.item(), episode + 1)
            writer.add_scalar('Loss/domain_loss_2', loss_disc.item(), episode + 1)
            # writer.add_scalar('Loss/text_align_loss', text_align_loss.item(), episode + 1)
            # writer.add_scalar('Loss/scl_loss_tar', scl_loss_tar.item(), episode + 1)
            writer.add_scalar('Loss/CTx_loss_tar', CTx_loss_tar.item(), episode + 1)
            writer.add_scalar('Loss/loss', total_loss.item(), episode + 1)

            writer.add_scalar('Acc/acc_src', acc_src, episode + 1)
            writer.add_scalar('Acc/acc_tar', acc_tar, episode + 1)

        if (episode + 1) % 500 == 0 or episode == 0:     # 定期测试评估
            with torch.no_grad():
                # 测试
                logger.info("Testing ...")
                train_end = time.time()
                # 切换到评估模式
                mapping_tar.eval()
                encoder.eval()
                total_rewards = 0         # 累计正确预测数
                counter = 0               # 累计测试样本数
                accuracies = []
                predict = np.array([], dtype=np.int64)
                predict_gnn = np.array([], dtype=np.int64)
                labels = np.array([], dtype=np.int64)

                full_predict = np.array([], dtype=np.int64)

                train_datas, train_labels = train_loader.__iter__().__next__()

                support_real_labels = train_labels

                train_features, _ = encoder(mapping_tar(Variable(train_datas).to(GPU)))           # 提取支持集特征

                # 特征归一化处理
                max_value = train_features.max()
                min_value = train_features.min()
                print(max_value.item())
                print(min_value.item())
                train_features = (train_features - min_value) * 1.0 / (max_value - min_value)

                #  KNN分类器构建
                KNN_classifier = KNeighborsClassifier(n_neighbors=1)
                KNN_classifier.fit(train_features.cpu().detach().numpy(), train_labels)
                for test_datas, test_labels in test_loader:
                    batch_size = test_labels.shape[0]

                    test_features, _ = encoder(mapping_tar((Variable(test_datas).to(GPU))))   # 提取测试特征
                    test_features = (test_features - min_value) * 1.0 / (max_value - min_value)     # 归一化
                    predict_labels = KNN_classifier.predict(test_features.cpu().detach().numpy())   # KNN预测
                    test_labels = test_labels.numpy()
                    rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)]

                    total_rewards += np.sum(rewards)
                    counter += batch_size

                    predict = np.append(predict, predict_labels)
                    labels = np.append(labels, test_labels)

                    accuracy = total_rewards / 1.0 / counter
                    accuracies.append(accuracy)

                test_accuracy = 100. * total_rewards / len(test_loader.dataset)
                writer.add_scalar('Acc/acc_test', test_accuracy, episode + 1)

                logger.info('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format(total_rewards, len(test_loader.dataset),
                                                                     100. * total_rewards / len(test_loader.dataset)))
                test_end = time.time()

                mapping_tar.train()
                encoder.train()

                if test_accuracy > last_accuracy:
                    last_accuracy = test_accuracy
                    best_episode = episode
                    acc[iDataSet] = 100. * total_rewards / len(test_loader.dataset)
                    OA = acc
                    C = metrics.confusion_matrix(labels, predict)
                    A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=float)
                    best_predict_all = predict
                    best_labels_all = labels  # 保存真实标签用于F1计算
                    best_G, best_RandPerm, best_Row, best_Column, best_nTrain = G, RandPerm, Row, Column, nTrain
                    k[iDataSet] = metrics.cohen_kappa_score(labels, predict)
                    # 计算F1-score (weighted average)
                    f1_scores[iDataSet] = metrics.f1_score(labels, predict, average='weighted')
                    
                    # 保存最佳模型参数（用于后续可视化）
                    best_mapping_src_state = copy.deepcopy(mapping_src.state_dict())
                    best_mapping_tar_state = copy.deepcopy(mapping_tar.state_dict())
                    best_encoder_state = copy.deepcopy(encoder.state_dict())
                    best_cross_transformer_state = copy.deepcopy(cross_transformer.state_dict())
                

                logger.info('best episode:[{}], best accuracy={}'.format(best_episode + 1, last_accuracy))


    logger.info('iter:{} best episode:[{}], best accuracy={}'.format(iDataSet, best_episode + 1, last_accuracy))
    logger.info("train time per DataSet(s): " + "{:.5f}".format(train_end - train_start))
    logger.info("accuracy list: {}".format(acc))
    logger.info('***********************************************************************************')

OAMean = np.mean(acc)
OAStd = np.std(acc)

AA = np.mean(A, 1)
AAMean = np.mean(AA, 0)
AAStd = np.std(AA)

kMean = np.mean(k)
kStd = np.std(k)

f1Mean = np.mean(f1_scores)
f1Std = np.std(f1_scores)

AMean = np.mean(A, 0)
AStd = np.std(A, 0)

logger.info("train time per DataSet(s): " + "{:.5f}".format(train_end - train_start))
logger.info("test time per DataSet(s): " + "{:.5f}".format(test_end - train_end))
logger.info("average OA: " + "{:.2f}".format(OAMean) + " +- " + "{:.2f}".format(OAStd))
logger.info("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
logger.info("average kappa: " + "{:.4f}".format(100 * kMean) + " +- " + "{:.4f}".format(100 * kStd))
logger.info("average F1-score: " + "{:.4f}".format(100 * f1Mean) + " +- " + "{:.4f}".format(100 * f1Std))
logger.info("accuracy list: {}".format(acc))
logger.info("accuracy for each class: ")
for i in range(TAR_CLASS_NUM):
    logger.info("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))

if args.run_once:
    # 供 bayesian_tuning_universal.py 解析的机器可读输出
    result_payload = {
        "best_accuracy": float(OAMean),
        "acc_per_seed": [float(x) for x in acc.reshape(-1).tolist()],
        "seeds": [int(s) for s in seeds],
        "lambda_weight": float(args.lambda_weight),
        "mask_ratio": float(args.mask_ratio),
        "weight_ctx": float(args.weight_ctx),
        "weight_wd": float(args.weight_wd),
        "weight_disc": float(args.weight_disc),
        "max_episode": int(EPISODE),
    }
    print("TUNE_RESULT_JSON=" + json.dumps(result_payload, ensure_ascii=False), flush=True)







