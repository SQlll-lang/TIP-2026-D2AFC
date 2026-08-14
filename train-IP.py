import numpy as np
import os
import argparse
import pickle
import time
import importlib.util
import random
import logging
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from model.mapping import Mapping
from utils.dataloader import get_HBKC_data_loader, Task, get_target_dataset, tagetSSLDataset
from utils import utils, data_augment
from model.GLAI_Former import GLAIFormer, CrossTransformer, DomainDiscriminator

from model import loss
from model.loss import ConTeXLoss


# import warnings

# warnings.simplefilter(action='ignore', category=FutureWarning)


parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument('--config', type=str, default=os.path.join('./config', 'IP.py'))
args = parser.parse_args()

# load hyperparameters
_spec = importlib.util.spec_from_file_location("exp_config", args.config)
_config_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_config_mod)
config = _config_mod.config
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
N_DIMENSION = train_opt['n_dim']                       # 目前还不清楚作用
SHOT_NUM_PER_CLASS = train_opt['shot_num_per_class']
QUERY_NUM_PER_CLASS = train_opt['query_num_per_class']
EPISODE = train_opt['episode']
LEARNING_RATE = train_opt['lr']
GPU = config['gpu']
TAR_CLASS_NUM = train_opt['tar_class_num']  # the number of class
TAR_LSAMPLE_NUM_PER_CLASS = train_opt['tar_lsample_num_per_class']  # 每个类别的标签样本数
SCR_CLASS_NUM = train_opt['scr_class_num']   # 源域样本数
WEIGHT_DECAY = train_opt['weight_decay']

utils.same_seeds(0)

# load source domain data
print('----------------------------load source domain data----------------------------')
with open(os.path.join(data_path, source_data), 'rb') as handle:    # 加载源域数据
    source_imdb = pickle.load(handle)

data_train = source_imdb['data']
labels_train = source_imdb['Labels']


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
Data_Band_Scaler, GroundTruth = utils.load_data_IP(test_data, test_label)  # 加载测试数据和标签

# loss init
crossEntropy = nn.CrossEntropyLoss().to(GPU)

wd_loss = loss.SinkhornDistance(eps=0.1, max_iter=100, reduction='mean')   # Wasserstein距离损失  用于全局分布域对齐
domain_criterion_2 = nn.BCEWithLogitsLoss().to(GPU)    # 二进制交叉熵损失 域判别损失 D2
contex_loss_fn = ConTeXLoss(temperature=0.1, lambda_weight=0.7).to(GPU)

# experimental result index
nDataSet = 10    # 定义运行次数
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, TAR_CLASS_NUM])   # 存储每次实验中每个类别的结果
k = np.zeros([nDataSet, 1])
f1_scores = np.zeros([nDataSet, 1])  # 存储F1-score
best_predict_all = []   # 用于存储所有实验中的最佳预测结果
best_predict_full = []
best_G, best_RandPerm, best_Row, best_Column, best_nTrain = None, None, None, None, None   # 存储实验过程中的最佳参数
best_full_G, best_Row_all, best_Column_all, best_nTrain_all = None, None, None, None
best_mapping_src_state, best_mapping_tar_state, best_encoder_state = None, None, None  # 存储最佳模型参数


seeds = [1228, 1237, 1226, 1227, 1211, 1212, 1241, 1240, 1222, 1223]

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

    target_ssl_dataset = tagetSSLDataset(target_aug_data_ssl, target_aug_label_ssl)      # 目标域数据加载器
    target_ssl_dataloader = torch.utils.data.DataLoader(target_ssl_dataset, batch_size=64, shuffle=True, drop_last=True)

    # 生成小样本学习任务所需的掩码和参数  一个小样本任务的支持集大小、总样本大小、查询集掩码
    num_supports, num_samples, query_edge_mask, evaluation_mask = utils.preprocess(TAR_CLASS_NUM, SHOT_NUM_PER_CLASS,
                                                                                   QUERY_NUM_PER_CLASS, batch_task, GPU)
    # 定义源域和目标域的映射网络（将原始特征映射到公共空间）与特征提取网络Encoder
    mapping_src = Mapping(SRC_INPUT_DIMENSION, N_DIMENSION).to(GPU)
    mapping_tar = Mapping(TAR_INPUT_DIMENSION, N_DIMENSION).to(GPU)

    cross_transformer = CrossTransformer(embed_dim=emb_size, num_heads=8) # 初始化 Cross-Transformer
    domain_classifier_2 = DomainDiscriminator().to(GPU)


    encoder = GLAIFormer(inp_channels=N_DIMENSION, dim=emb_size, depths=[1], num_heads_spa=[8], num_heads_spe=[7], dropout=0.3).to(GPU)

    mapping_src_optim = torch.optim.SGD(mapping_src.parameters(), lr=LEARNING_RATE, momentum=0.9,
                                        weight_decay=WEIGHT_DECAY)
    mapping_tar_optim = torch.optim.SGD(mapping_tar.parameters(), lr=LEARNING_RATE, momentum=0.9,
                                        weight_decay=WEIGHT_DECAY)
    encoder_optim = torch.optim.SGD(encoder.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)

    cross_transformer_optim = torch.optim.SGD(cross_transformer.parameters(), lr=LEARNING_RATE, momentum=0.9,
                                              weight_decay=WEIGHT_DECAY)
    domain_classifier_2_optim = torch.optim.SGD(domain_classifier_2.parameters(), lr=LEARNING_RATE, momentum=0.9,
                                              weight_decay=WEIGHT_DECAY)


    # 应用自定义权重初始化
    mapping_src.apply(utils.weights_init)
    mapping_tar.apply(utils.weights_init)
    domain_classifier_2.apply(utils.weights_init)

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
    domain_classifier_2.train()
    encoder.train()
    cross_transformer.train()

    logger.info("Training...")
    last_accuracy = 0.0                # 记录上一次准确率
    best_episode = 0                   # 最佳训练轮次
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


        # 源域特征提取
        support_features_src, support_output_src = encoder(mapping_src(support_src.to(GPU)))  # (14, 128)
        query_features_src, query_output_src = encoder(mapping_src(query_src.to(GPU)))            # (266, 128)

        # 目标域特征提取（同理）
        support_features_tar, support_output_tar = encoder(mapping_tar(support_tar.to(GPU)))  # (14, 128)    包含一个语义特征的映射过程
        query_features_tar, query_output_tar = encoder(mapping_tar(query_tar.to(GPU)))   # torch.Size([266, 128])


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
            data_augment.random_mask_batch_spatial(target_ssl_data.data.cpu(), 0.8))  # (64, 150, 7, 7)
        augment2_target_ssl_data = torch.FloatTensor(
            data_augment.random_mask_batch_spatial(target_ssl_data.data.cpu(), 0.8))  # (64, 150, 7, 7)
        augment_target_ssl_data = torch.cat((augment1_target_ssl_data, augment2_target_ssl_data),
                                            dim=0)  #````````````````````````````````````````` ([128, 150, 7, 7])
        features_augment, _ = encoder(mapping_tar(augment_target_ssl_data.to(GPU)))  # (128, 128)   特征提取

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

        # 特征提取
        wd_s, _ = encoder(mapping_src(source_data.to(GPU)))  # (64, 128)   特征提取
        wd_t, _ = encoder(mapping_tar(target_data.to(GPU)))  # (64, 128)   特征提取
        loss_wd, _, _ = wd_loss(wd_s, wd_t)

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


        # 总损失
        loss = f_loss + 2.0 * CTx_loss_tar + 0.001 * loss_wd + 1 * loss_disc


        mapping_src.zero_grad()
        mapping_tar.zero_grad()
        encoder.zero_grad()
        domain_classifier_2_optim.zero_grad()
        cross_transformer_optim.zero_grad()

        loss.backward()   # 反向传播

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
                    loss_wd.item(),
                    loss_disc.item(),
                    CTx_loss_tar.item(),
                    loss.item(),
                    acc_src,
                    acc_tar))

            writer.add_scalar('Loss/f_loss', f_loss.item(), episode + 1)

            writer.add_scalar('Loss/loss_wd', loss_wd.item(), episode + 1)
            writer.add_scalar('Loss/domain_loss_2', loss_disc.item(), episode + 1)
            writer.add_scalar('Loss/CTx_loss_tar', CTx_loss_tar.item(), episode + 1)
            writer.add_scalar('Loss/loss', loss.item(), episode + 1)

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
                    best_G, best_RandPerm, best_Row, best_Column, best_nTrain = G, RandPerm, Row, Column, nTrain
                    k[iDataSet] = metrics.cohen_kappa_score(labels, predict)
                    # 计算F1-score (weighted average)
                    f1_scores[iDataSet] = metrics.f1_score(labels, predict, average='weighted')

                    # 保存最佳模型参数（用于后续可视化）
                    best_mapping_src_state = mapping_src.state_dict()
                    best_mapping_tar_state = mapping_tar.state_dict()
                    best_encoder_state = encoder.state_dict()


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









