import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

class Task(object):            # 在少样本学习任务中创建支持集（support set）和查询集（query set）
    def __init__(self, data, num_classes, shot_num, query_num):
        self.data = data
        self.num_classes = num_classes
        self.support_num = shot_num
        self.query_num = query_num

        class_folders = sorted(list(data))
        class_list = random.sample(class_folders, self.num_classes)   # 从所有类别中随机选择 num_classes 个类别，并将它们保存在 class_list 中
        labels = np.array(range(len(class_list)))
        labels = dict(zip(class_list, labels))
        samples = dict()

        self.support_datas = []
        self.query_datas = []
        self.support_labels = []
        self.query_labels = []

        self.support_real_labels = []    # 存储支持集样本的原始类别标签（即类别名）
        self.query_real_labels = []      # 存储查询集样本的原始类别标签（即类别名
        for c in class_list:
            temp = self.data[c]  # list
            samples[c] = random.sample(temp, len(temp))
            random.shuffle(samples[c])

            self.support_datas += samples[c][:shot_num]
            self.query_datas += samples[c][shot_num:shot_num + query_num]

            self.support_labels += [labels[c] for i in range(shot_num)]
            self.query_labels += [labels[c] for i in range(query_num)]

            self.support_real_labels += [c for i in range(shot_num)]
            self.query_real_labels += [c for i in range(query_num)]

class FewShotDataset(Dataset):
    def __init__(self, task, split='train'):
        self.task = task
        self.split = split
        self.image_datas = self.task.support_datas if self.split == 'train' else self.task.query_datas
        self.labels = self.task.support_labels if self.split == 'train' else self.task.query_labels

    def __len__(self):
        return len(self.image_datas)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")

class HBKC_dataset(FewShotDataset):
    def __init__(self, *args, **kwargs):
        super(HBKC_dataset, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image = self.image_datas[idx]
        label = self.labels[idx]
        return image, label

# 解决在分类任务中不同类别样本数量不平衡的问题，确保每个类别的样本数在每次加载时都是相同的
class ClassBalancedSampler(Sampler):
    def __init__(self, num_per_class, num_cl, num_inst,shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        else:
            batch = [[i+j*self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1

def get_HBKC_data_loader(task, num_per_class=1, split='train',shuffle = False):     # 创建用于训练或测试的DataLoader
    dataset = HBKC_dataset(task, split=split)
    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.support_num, shuffle=shuffle) # support set
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.query_num, shuffle=shuffle) # query set
    loader = DataLoader(dataset, batch_size=num_per_class*task.num_classes, sampler=sampler)
    return loader

class MetaTrainLabeledDataset(Dataset):
    def __init__(self, image_datas, image_labels):
        self.image_datas = image_datas
        self.image_labels = image_labels

    def __len__(self):
        return len(self.image_datas)

    def __getitem__(self, idx):
        image = self.image_datas[idx]
        label = self.image_labels[idx]
        return image, label



from . import utils, data_augment
import math

def get_allpixel_loader(Data_Band_Scaler, GroundTruth,  HalfWidth):

    # 原图尺寸与波段数
    nRow, nColumn, nBand = Data_Band_Scaler.shape
    print(f'Input H×W×B = {nRow}×{nColumn}×{nBand}')

    # 翻转以便做镜像填充
    data_pad = utils.flip(Data_Band_Scaler)
    gt_pad   = utils.flip(GroundTruth)

    # 从翻转后图像中截取带半宽镜像填充的区域
    G = gt_pad[
        nRow - HalfWidth : 2 * nRow + HalfWidth,
        nColumn - HalfWidth : 2 * nColumn + HalfWidth
    ]  # shape: (nRow+2*HalfWidth, nColumn+2*HalfWidth)
    data = data_pad[
        nRow - HalfWidth : 2 * nRow + HalfWidth,
        nColumn - HalfWidth : 2 * nColumn + HalfWidth,
        :
    ]  # shape: (nRow+2*HalfWidth, nColumn+2*HalfWidth, nBand)

    # 我们要预测的位置：中心点在原图区域的所有像素（包含背景）
    rows = np.arange(HalfWidth, HalfWidth + nRow)
    cols = np.arange(HalfWidth, HalfWidth + nColumn)
    Row, Column = np.meshgrid(rows, cols, indexing='ij')
    Row = Row.ravel()
    Column = Column.ravel()
    nSample = Row.size
    print(f'Number of total pixels (incl. background): {nSample}')

    # 准备存放补丁和标签
    P = 2 * HalfWidth + 1
    trainX = np.zeros((nSample, P, P, nBand), dtype=np.float32)
    trainY = np.zeros((nSample,),      dtype=np.int64)

    # 逐像素抽补丁，标签直接取 G 中的值（0 表示背景）
    for idx in range(nSample):
        r, c = Row[idx], Column[idx]
        patch = data[
            r - HalfWidth : r + HalfWidth + 1,
            c - HalfWidth : c + HalfWidth + 1,
            :
        ]
        trainX[idx] = patch
        trainY[idx] = G[r, c]  # 0…class_num

    # 转成 (N, C, H, W) 并把标签从 0…N_cls 变成 0…N_cls（保持不动，背景仍是 0）
    trainX = trainX.transpose(0, 3, 1, 2)
    # （如果你的网络期望标签从 0 开始且类别数 = class_num），这里就不需要减 1：
    # trainY = trainY  # background=0, classes 1…class_num

    print('all data shape:', trainX.shape)
    print('all label shape:', trainY.shape)

    # 构造 DataLoader
    test_dataset = MetaTrainLabeledDataset(image_datas=trainX, image_labels=trainY)
    test_loader  = DataLoader(
        test_dataset,
        batch_size=100, shuffle=False
    )
    del test_dataset

    # 返回：DataLoader、带 padding 的标签图 G，用于可视化时定位坐标，以及总样本数
    return test_loader, G, Row, Column, nSample


def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, tar_lsample_num_per_class, shot_num_per_class, HalfWidth):

    print(Data_Band_Scaler.shape)    # 打印输入数据维度，例如：(H, W, C)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    '''label start'''
    num_class = int(np.max(GroundTruth))    # 获取类别数
    data_band_scaler = utils.flip(Data_Band_Scaler)  # 数据镜像扩展（应对边缘采样）   填充操作
    groundtruth = utils.flip(GroundTruth)   # 标签同步镜像扩展
    del Data_Band_Scaler
    del GroundTruth

    # 截取中心区域（原数据被镜像后，实际取中间原始尺寸+边界扩展）
    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth, :]

    [Row, Column] = np.nonzero(G)    # 获取所有有效样本坐标（非背景像素）
    del data_band_scaler
    del groundtruth

    nSample = np.size(Row)     # 计算有效样本总数
    print('number of sample', nSample)

    # Sampling samples
    train = {}     # 存储每类训练样本索引
    test = {}      # 存储每类测试样本索引
    da_train = {}  # 数据增强后的训练样本索引
    m = int(np.max(G))                    # 实际最大类别号
    nlabeled = tar_lsample_num_per_class             # 目标域每类标记样本数

    # 计算数据增强倍数
    print('labeled number per class:', nlabeled)
    print((200 - nlabeled) / nlabeled + 1)
    print(math.ceil((200 - nlabeled) / nlabeled) + 1)    # 实际增强倍数公式

    # 按类别循环划分样本
    for i in range(m):
        # 获取属于当前类别的所有样本索引
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)    # 随机打乱
        nb_val = shot_num_per_class   # 每类取前nb_val个作为训练   这里的shot_num_per_class是主文件中定义的TAR_LSAMPLE_NUM_PER_CLASS
        train[i] = indices[:nb_val]   # 训练样本
        da_train[i] = []              # 初始化增强容器
        # 数据增强：通过重复样本实现
        for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1):
            da_train[i] += indices[:nb_val]      # 多次添加原始样本
        test[i] = indices[nb_val:]    # 剩余作为测试

    # 合并各类索引
    train_indices = []
    test_indices = []
    da_train_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
        da_train_indices += da_train[i]
    np.random.shuffle(test_indices)   # 打乱测试集顺序

    print('the number of train_indices:', len(train_indices))
    print('the number of test_indices:', len(test_indices))
    print('the number of train_indices after data argumentation:', len(da_train_indices))  # 增强后的样本数
    print('labeled sample indices:', train_indices)

    # 统计样本数
    nTrain = len(train_indices)
    nTest = len(test_indices)
    da_nTrain = len(da_train_indices)

    # 创建数据字典
    imdb = {}
    # 初始化数据容器：图像块大小(2H+1)x(2H+1)x波段数 x 总样本数
    imdb['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest],
                            dtype=np.float32)
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)   # 标签
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)     # 集合标识（1=训练，3=测试）

    # 合并索引并填充数据
    RandPerm = train_indices + test_indices

    RandPerm = np.array(RandPerm)

    # 逐个样本提取图像快
    for iSample in range(nTrain + nTest):
        imdb['data'][:, :, :, iSample] = data[
                                         Row[RandPerm[iSample]] - HalfWidth : Row[RandPerm[iSample]] + HalfWidth + 1,
                                         Column[RandPerm[iSample]] - HalfWidth : Column[RandPerm[iSample]] + HalfWidth + 1,
                                         :]   # 提取以当前坐标为中心的图像块
        imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)

    imdb['Labels'] = imdb['Labels'] - 1    # 标签从0开始（原G中类别从1开始）
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)    # 集合标识（1=训练，3=测试）
    print('Data is OK.')

    # 创建PyTorch数据集
    train_dataset = utils.matcifar(imdb, train=True, d=3, medicinal=0)    # 根据索引提取训练数据
    # 创建PyTorch数据集    批量大小=类别数×每类样本数（小样本常用设置）
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=class_num * shot_num_per_class, shuffle=False)
    del train_dataset

    test_dataset = utils.matcifar(imdb, train=False, d=3, medicinal=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)
    del test_dataset
    del imdb

    # 目标域用于训练的目标增强部分
    imdb_da_train = {}
    imdb_da_train['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain],
                                     dtype=np.float32)
    imdb_da_train['Labels'] = np.zeros([da_nTrain], dtype=np.int64)
    imdb_da_train['set'] = np.zeros([da_nTrain], dtype=np.int64)

    # # 应用辐射噪声增强
    da_RandPerm = np.array(da_train_indices)
    for iSample in range(da_nTrain):
        imdb_da_train['data'][:, :, :, iSample] = data_augment.radiation_noise(
            data[Row[da_RandPerm[iSample]] - HalfWidth: Row[da_RandPerm[iSample]] + HalfWidth + 1,
            Column[da_RandPerm[iSample]] - HalfWidth: Column[da_RandPerm[iSample]] + HalfWidth + 1, :])
        imdb_da_train['Labels'][iSample] = G[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)
    imdb_da_train['Labels'] = imdb_da_train['Labels'] - 1     # 同样调整标签起始
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64)     # 全部标记为训练集
    print('ok')

    # G表示填充后的图像大小    RandPerm表示位置索引    nTrain表示训练样本数
    return train_loader, test_loader, imdb_da_train, G, RandPerm, Row, Column, nTrain


def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, tar_lsample_num_per_class, shot_num_per_class, patch_size):
    train_loader, test_loader, imdb_da_train, G, RandPerm, Row, Column, nTrain = get_train_test_loader(   # 获取基础数据加载器和增强数据
        Data_Band_Scaler=Data_Band_Scaler,
        GroundTruth=GroundTruth,
        class_num=class_num,
        tar_lsample_num_per_class=tar_lsample_num_per_class,
        shot_num_per_class=shot_num_per_class,
        HalfWidth=patch_size // 2)
    train_datas, train_labels = train_loader.__iter__().__next__()
    print('train labels:', train_labels)
    print('size of train datas:', train_datas.shape)

    print(imdb_da_train.keys())            # 输出['data', 'Labels', 'set']
    print(imdb_da_train['data'].shape)
    print(imdb_da_train['Labels'])         # 验证增强后的标签序列

    del Data_Band_Scaler, GroundTruth

    # target data with data augmentation
    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))  # 换坐标轴 (9,9,103, 1800)->(1800, 103, 9, 9)
    print(target_da_datas.shape)
    target_da_labels = imdb_da_train['Labels']
    print('target data augmentation label:', target_da_labels)

    target_da_train_set = {}      # 按类别组织的字典 {类1:[样本1,样本2], 类2:[...]}
    target_aug_data_ssl = []      # 平铺结构增强数据样本列表   (N, C, H, W) 的平铺数组
    target_aug_label_ssl = []     # 平铺结构增强数据标签列表   (N,) 的平铺标签

    # 遍历每个增强样本
    for class_, path in zip(target_da_labels, target_da_datas):
        # 按类别组织字典
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
        # 同时构建平铺列表
        target_aug_data_ssl.append(path)
        target_aug_label_ssl.append(class_)
    target_da_metatrain_data = target_da_train_set
    print(target_da_metatrain_data.keys())     # 打印存在的类别键

    # target domain : batch samples for domian adaptation
    print(imdb_da_train['data'].shape)
    print(imdb_da_train['Labels'])
    target_dataset = utils.matcifar(imdb_da_train, train=True, d=3, medicinal=0)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=128, shuffle=True, num_workers=0, drop_last=True)
    del target_dataset


    return train_loader, test_loader, target_da_metatrain_data, target_loader, G, RandPerm, Row, Column, nTrain, target_aug_data_ssl, target_aug_label_ssl

def get_target_dataset_houston(Data_Band_Scaler, GroundTruth_train, GroundTruth_test, class_num, tar_lsample_num_per_class, shot_num_per_class, patch_size):
    train_loader, _, imdb_da_train, _, _, _, _, _ = get_train_test_loader(
        Data_Band_Scaler=Data_Band_Scaler,
        GroundTruth=GroundTruth_train,
        class_num=class_num,
        tar_lsample_num_per_class=tar_lsample_num_per_class,
        shot_num_per_class=shot_num_per_class,
        HalfWidth=patch_size // 2)
    test_loader, G, RandPerm, Row, Column, nTrain = get_alltest_loader(
        Data_Band_Scaler=Data_Band_Scaler,
        GroundTruth=GroundTruth_test,
        class_num=class_num,
        shot_num_per_class=0,
        HalfWidth=patch_size // 2)


    train_datas, train_labels = train_loader.__iter__().__next__()
    print('train labels:', train_labels)
    print('size of train datas:', train_datas.shape)

    print(imdb_da_train.keys())
    print(imdb_da_train['data'].shape)
    print(imdb_da_train['Labels'])
    del Data_Band_Scaler, GroundTruth_train, GroundTruth_test

    # target data with data augmentation
    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))  # 换坐标轴 (9,9,103, 1800)->(1800, 103, 9, 9)
    print(target_da_datas.shape)
    target_da_labels = imdb_da_train['Labels']
    print('target data augmentation label:', target_da_labels)

    # metatrain data for few-shot classification 和之前的区别就是，把多维数组按类别划分为字典。
    target_da_train_set = {}
    target_aug_data_ssl = []
    target_aug_label_ssl = []

    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
        target_aug_data_ssl.append(path)
        target_aug_label_ssl.append(class_)
    target_da_metatrain_data = target_da_train_set
    print(target_da_metatrain_data.keys())

    return train_loader, test_loader, target_da_metatrain_data, G, RandPerm, Row, Column, nTrain, target_aug_data_ssl, target_aug_label_ssl


def get_alltest_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class, HalfWidth):

    print(Data_Band_Scaler.shape)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    '''label start'''
    num_class = int(np.max(GroundTruth))
    data_band_scaler = utils.flip(Data_Band_Scaler)
    groundtruth = utils.flip(GroundTruth)

    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth,
        nColumn - HalfWidth:2 * nColumn + HalfWidth]
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth,
           :]

    [Row, Column] = np.nonzero(G)
    del data_band_scaler
    del groundtruth

    nSample = np.size(Row)
    max_Row = np.max(Row)
    print('number of sample', nSample)

    train = {}

    m = int(np.max(G))

    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        nb_val = int(len(indices))
        train[i] = indices[:nb_val]

    train_indices = []

    for i in range(m):
        train_indices += train[i]
    np.random.shuffle(train_indices)

    print('the number of target:', len(train_indices))

    nTrain = len(train_indices)

    trainX = np.zeros([nTrain,  2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand], dtype=np.float32)
    trainY = np.zeros([nTrain], dtype=np.int64)

    RandPerm = train_indices
    RandPerm = np.array(RandPerm)
    for i in range(nTrain):
        trainX[i, :, :, :] = data[Row[RandPerm[i]] - HalfWidth: Row[RandPerm[i]] + HalfWidth + 1, Column[RandPerm[i]] - HalfWidth: Column[RandPerm[i]] + HalfWidth + 1, :] # 7 7 144
        trainY[i] = G[Row[RandPerm[i]], Column[RandPerm[i]]].astype(np.int64)
    trainX = np.transpose(trainX, (0, 3, 1, 2))
    trainY = trainY - 1

    print('all data shape', trainX.shape)
    print('all label shape', trainY.shape)

    test_dataset = MetaTrainLabeledDataset(image_datas=trainX, image_labels=trainY)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
    return test_loader, G, RandPerm, Row, Column, nTrain


class tagetSSLDataset(Dataset):
    def __init__(self, image_datas, image_labels):
        self.image_datas = image_datas
        self.image_labels = image_labels

    def __len__(self):
        return len(self.image_datas)

    def __getitem__(self, idx):
        image = self.image_datas[idx]
        label = self.image_labels[idx]
        return image, label






