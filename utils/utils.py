import numpy as np
import random
import scipy.io as sio
from sklearn import preprocessing
import matplotlib.pyplot as plt
import os
import logging
import datetime

import torch
import torch.nn as nn

def same_seeds(seed):   # 设置随机数生成的种子，以确保在不同的运行中得到相同的结果
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.    # 确保NumPy在生成随机数时也使用相同的种子
    random.seed(seed)  # Python random module.     # 确保在Python标准库中的随机数生成器也使用给定的种子值
    torch.backends.cudnn.benchmark = False   # 告诉cuDNN（NVIDIA的深度学习库）不要优化针对特定硬件的自动调优策略，从而使得训练过程具有可重复性
    torch.backends.cudnn.deterministic = True  # cuDNN将使用确定性算法来执行卷积操作，避免由于硬件和并行计算的不同引起的结果波动

def weights_init(m):
    if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
        if m.weight is not None:
            nn.init.normal_(m.weight, 1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())

import torch.utils.data as data

class matcifar(data.Dataset):
    def __init__(self, imdb, train, d, medicinal):    # d：一个控制数据转置维度的参数  medicinal：数据加载方式

        self.train = train  # training set or test set
        self.imdb = imdb
        self.d = d
        self.x1 = np.argwhere(self.imdb['set'] == 1)   # 值为 1（训练集）和值为 3（测试机）的位置索引
        self.x2 = np.argwhere(self.imdb['set'] == 3)
        self.x1 = self.x1.flatten()
        self.x2 = self.x2.flatten()

        if medicinal == 1:
            self.train_data = self.imdb['data'][self.x1, :, :, :]
            self.train_labels = self.imdb['Labels'][self.x1]
            self.test_data = self.imdb['data'][self.x2, :, :, :]
            self.test_labels = self.imdb['Labels'][self.x2]

        else:
            self.train_data = self.imdb['data'][:, :, :, self.x1]
            self.train_labels = self.imdb['Labels'][self.x1]
            self.test_data = self.imdb['data'][:, :, :, self.x2]
            self.test_labels = self.imdb['Labels'][self.x2]
            if self.d == 3:
                self.train_data = self.train_data.transpose((3, 2, 0, 1))
                self.test_data = self.test_data.transpose((3, 2, 0, 1))
            else:
                self.train_data = self.train_data.transpose((3, 0, 2, 1))
                self.test_data = self.test_data.transpose((3, 0, 2, 1))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:

            img, target = self.train_data[index], self.train_labels[index]
        else:

            img, target = self.test_data[index], self.test_labels[index]

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


def sanity_check(all_set):   # 检查传入的数据集，并对数据进行处理  确保每个类别至少有200个样本,如果超过保存最后的200个
    nclass = 0
    nsamples = 0
    all_good = {}
    for class_ in all_set:
        if len(all_set[class_]) >= 200:
            all_good[class_] = all_set[class_][len(all_set[class_])-200:]
            nclass += 1
            nsamples += len(all_good[class_])
    print('the number of class:', nclass)
    print('the number of sample:', nsamples)
    return all_good

def flip(data):    # 对输入数据进行特定的拼接操作，生成一个新的数据数组
    y_4 = np.zeros_like(data)
    y_1 = y_4
    y_2 = y_4
    first = np.concatenate((y_1, y_2, y_1), axis=1)
    second = np.concatenate((y_4, data, y_4), axis=1)
    third = first
    Data = np.concatenate((first, second, third), axis=0)
    return Data


def load_data_custom_GF51105(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)

    data_all = image_data['image_data']
    data_all = data_all.transpose(1, 2, 0)     # 仅在此数据集上
    GroundTruth = label_data['image_data']

    # data_all = image_data['indian_pines_corrected']
    # GroundTruth = label_data['indian_pines_gt']

    # data_all = image_data['Houston']
    # GroundTruth = label_data['Houston_gt']

    [nRow, nColumn, nBand] = data_all.shape
    print('GF51105', nRow, nColumn, nBand)

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    data_scaler = preprocessing.scale(data)  # (X-X_mean)/X_std,
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1],data_all.shape[2])

    return Data_Band_Scaler, GroundTruth

def load_data_custom_ZY10424(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)

    data_all = image_data['image_data']
    data_all = data_all.transpose(1, 2, 0)     # 仅在此数据集上
    GroundTruth = label_data['image_data']

    # data_all = image_data['indian_pines_corrected']
    # GroundTruth = label_data['indian_pines_gt']

    # data_all = image_data['Houston']
    # GroundTruth = label_data['Houston_gt']

    [nRow, nColumn, nBand] = data_all.shape
    print('ZY10424', nRow, nColumn, nBand)

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    data_scaler = preprocessing.scale(data)  # (X-X_mean)/X_std,
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1],data_all.shape[2])

    return Data_Band_Scaler, GroundTruth

def load_data_houston(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)

    # data_all = image_data['indian_pines_corrected']
    # GroundTruth = label_data['indian_pines_gt']

    data_all = image_data['Houston']
    GroundTruth = label_data['Houston_gt']

    [nRow, nColumn, nBand] = data_all.shape
    print('Houston', nRow, nColumn, nBand)

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    data_scaler = preprocessing.scale(data)  # (X-X_mean)/X_std,
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1],data_all.shape[2])

    return Data_Band_Scaler, GroundTruth


def load_data_IP(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)

    data_all = image_data['indian_pines_corrected']
    GroundTruth = label_data['indian_pines_gt']

    # data_all = image_data['Houston']
    # GroundTruth = label_data['Houston_gt']

    [nRow, nColumn, nBand] = data_all.shape
    print('Houston', nRow, nColumn, nBand)

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    data_scaler = preprocessing.scale(data)  # (X-X_mean)/X_std,
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1],data_all.shape[2])

    return Data_Band_Scaler, GroundTruth

def load_data_GFYC(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)

    # data_all = image_data['indian_pines_corrected']
    # GroundTruth = label_data['indian_pines_gt']

    data_all = image_data['Data']
    GroundTruth = label_data['DataClass']

    [nRow, nColumn, nBand] = data_all.shape
    print('GFYC', nRow, nColumn, nBand)

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    data_scaler = preprocessing.scale(data)  # (X-X_mean)/X_std,
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1],data_all.shape[2])

    return Data_Band_Scaler, GroundTruth

def load_data_GF50525(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)

    # data_all = image_data['indian_pines_corrected']
    # GroundTruth = label_data['indian_pines_gt']

    data_all = image_data['image_data']
    data_all = data_all.transpose(1, 2, 0)
    GroundTruth = label_data['DataClass']

    [nRow, nColumn, nBand] = data_all.shape
    print('GF50525', nRow, nColumn, nBand)

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    data_scaler = preprocessing.scale(data)  # (X-X_mean)/X_std,
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1],data_all.shape[2])

    return Data_Band_Scaler, GroundTruth

def compute_features(device, model, mapping_tar, loader):
    """
    从 loader 中按 batch 提取特征并返回 numpy arrays: (N, D), (N,)
    假设 model(mapping_tar(x)) 返回 (features, other) 或 直接返回 features。
    """
    model.eval()
    mapping_tar.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # inputs -> device
            inputs = inputs.to(device)

            # forward
            out, _ = model(mapping_tar(inputs))
            # out 可能是 tensor，也可能是 tuple/list (features, ...)
            if isinstance(out, (tuple, list)):
                feature_tensor = out[0]
            else:
                feature_tensor = out

            # 确保是 torch.Tensor
            if not isinstance(feature_tensor, torch.Tensor):
                raise TypeError(f"Expected feature tensor, got {type(feature_tensor)}")

            # detach + move to cpu + numpy
            feat_np = feature_tensor.detach().cpu().numpy()  # shape: (batch, ...)

            # 展平每个样本为向量 (batch, D)
            feat_np = feat_np.reshape(feat_np.shape[0], -1)
            features_list.append(feat_np)

            # targets 可能在 cpu 或 gpu，统一处理为 numpy
            if isinstance(targets, torch.Tensor):
                labels_list.append(targets.detach().cpu().numpy())
            else:
                # 如果 loader 返回 ndarray/list
                labels_list.append(np.array(targets))

    # 拼接所有 batch
    if len(features_list) == 0:
        return np.zeros((0, 0)), np.zeros((0,), dtype=int)

    features = np.vstack(features_list)  # (N, D)
    labels = np.concatenate(labels_list)  # (N,)

    return features, labels

def classification_map(map, groundTruth, dpi, savePath):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(groundTruth.shape[1]*2.0/dpi, groundTruth.shape[0]*2.0/dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(map)
    fig.savefig(savePath, dpi = dpi)

    return 0

def preprocess(num_ways, num_shots, num_queries, batch_size, device):    # 处理少样本学习任务中的支持集（support set）和查询集（query set）
    """
    prepare for train and evaluation
    :param num_ways: number of classes for each few-shot task
    :param num_shots: number of samples for each class in few-shot task    支持集样本数
    :param num_queries: number of queries for each class in few-shot task  查询集样本数
    :param batch_size: how many tasks per batch
    :param device: the gpu device that holds all data
    :return: number of samples in support set
             number of total samples (support and query set)
             mask for edges connect query nodes
             mask for unlabeled data (for semi-supervised setting)
    """
    # set size of support set, query set and total number of data in single task
    num_supports = num_ways * num_shots # 14 * 1 = 14
    num_samples = num_supports + num_queries * num_ways #  9 * 1 + 19 * 15   计算样本总数

    # set edge mask (to distinguish support and query edges) 设置边掩码（用于区分支持和查询边）
    support_edge_mask = torch.zeros(batch_size, num_samples, num_samples).to(device)
    support_edge_mask[:, :num_supports, :num_supports] = 1
    query_edge_mask = 1 - support_edge_mask            # 查询集掩码  用1表示查询集样本，0表示支持集样本
    evaluation_mask = torch.ones(batch_size, num_samples, num_samples).to(device) # 未标记样本的掩码 (用于半监督学习)
    return num_supports, num_samples, query_edge_mask, evaluation_mask

def set_logging_config(logdir, num_seeds):   # 配置日志记录系统
    myTimeFormat = '%Y-%m-%d_%H-%M-%S'     # 定义时间戳格式
    nowTime = datetime.datetime.now().strftime(myTimeFormat)   # 获取当前时间

    if not os.path.exists(logdir):   # 如果不存在，创建日志文件
        os.makedirs(logdir)
    logging.basicConfig(format="[%(asctime)s] [%(levelname)s] %(message)s",
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(logdir, str(num_seeds) +'seeds_'+nowTime+'.log')),
                                  logging.StreamHandler(os.sys.stdout)])

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits


def prototype_rectification(support_features, query_features, initial_prototypes):
    """
    Args:
        support_features (Tensor): 支持集特征, shape [N_way, K_shot, feature_dim]
        query_features (Tensor): 查询集特征, shape [N_way, N_query, feature_dim]
    Returns:
        rectified_prototypes (Tensor): 修正后的原型, shape [N_way, feature_dim]
    """

    # 计算支持集和查询集的整体均值差异作为偏移项δ
    support_mean = torch.mean(support_features, dim=(1, 0))  # 全局均值 [feature_dim]
    query_mean = torch.mean(query_features, dim=(1, 0))  # 全局均值 [feature_dim]
    delta = query_mean - support_mean

    # 更新原型：P_c^new = P_c + δ
    rectified_prototypes = initial_prototypes + delta.unsqueeze(0)
    return rectified_prototypes
