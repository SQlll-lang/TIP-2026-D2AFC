import torch

torch.set_printoptions(profile="full")
import torch.nn as nn
import numpy as np

import torch.nn.functional as F

# 子域分布对齐
class stMMD_loss(nn.Module):  # 综合了子域–最大均值差（subdomain MMD）、对比学习（contrastive loss）等策略，用于增强域适应时的类别对齐
    def __init__(self, class_num=31, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, temperature=0.07,
                 base_temperature=0.07):
        super(stMMD_loss, self).__init__()
        self.class_num = class_num
        # 控制 Gaussian 核的参数，多尺度 RBF 核合并以更好地度量分布差异
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type
        # 对比学习中的温度缩放系数，影响 logits 的平滑度
        self.temperature = temperature
        self.base_temperature = base_temperature

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):  # 多尺度 Gaussian 核函数
        # 对齐子域分布时，需计算源+目标所有样本对的核值矩阵，后面可从中切片得到 SS/TT/ST/TS 四个子矩阵
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:  # 未设置fix_sigma，则由所有成对距离的平均值自适应设定
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def source_guassian_kernel(self, source, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        # 与上面类似，但只接受单一输入 source，返回 (N×N) 的核矩阵，主要用于对比损失里把所有样本两两度量
        n_samples = int(source.size()[0])
        total = source
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def get_contrast_loss(self, source, s_label):  # 对比学习损失
        # 以 kernel 值作为“相似度”，推动同类样本在特征空间内更靠近（增大 kernel），异类更远离
        batch_size = source.shape[0]
        # 1) 计算 N×N 的多尺度 RBF 核相似度矩阵
        kernels = self.source_guassian_kernel(source, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                              fix_sigma=self.fix_sigma)
        if torch.sum(torch.isnan(sum(kernels))): return torch.tensor([0.]).cuda()
        # 构造“同类”掩码： mask[i,j]=1 当且仅当 s_label[i]==s_label[j]
        s_label = s_label.contiguous().view(-1, 1)
        mask = torch.eq(s_label, s_label.T).float().cuda()
        # 3) 将核值当作对比 logits，并做温度缩放
        anchor_dot_contrast = torch.div(kernels, self.temperature * self.kernel_num)
        # 4) 为数值稳定性，减去每行最大的值
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 5) 掩去自身对比（i==j）
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask

        # 6) 计算 log‑probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # 7) 只对那些至少有一个正样本的行参与计算
        non_zero_mask = mask.sum(1) != 0
        # # 每个样本与同类的平均 log‑prob
        mean_log_prob_pos = (mask[non_zero_mask] * log_prob[non_zero_mask]).sum(1) / mask[non_zero_mask].sum(1)

        # 8) 最终 loss：负号 + 温度缩放比
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def source_guassian_kernel2(self, source, kernel_mul=2.0, kernel_num=5, fix_sigma=None,
                                temperature=1.):  # 与上面类似，更便于多尺度对比学习
        n_samples = int(source.size()[0])
        total = source
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernels = [-L2_distance / bandwidth_temp for bandwidth_temp in bandwidth_list]
        # kernel_vals = [-L2_distance / bandwidth_temp / temperature for bandwidth_temp in bandwidth_list]
        # kernel_val = [_kernel_val - torch.max(_kernel_val,dim=1,keepdim=True)[0].detach() for _kernel_val in kernel_vals]
        # kernel_val = [torch.exp(_kernel_val) for _kernel_val in kernel_vals]
        # return sum(kernel_val)
        return kernels

    def get_contrast_loss2(self, source,
                           s_label):  # 与 get_contrast_loss 类似，但对每个尺度的 kernel 分别计算一次 loss，然后取平均；便于研究不同尺度的对比影响
        batch_size = source.shape[0]
        kernels = self.source_guassian_kernel2(source, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                               fix_sigma=self.fix_sigma)
        if torch.sum(torch.isnan(sum(kernels))): return torch.tensor([0.]).cuda()
        s_label = s_label.contiguous().view(-1, 1)
        mask = torch.eq(s_label, s_label.T).float().cuda()
        t_loss = []
        for kernel in kernels:
            # compute logits
            # anchor_dot_contrast = kernels
            anchor_dot_contrast = kernel
            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            # mask-out self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size).view(-1, 1).cuda(),
                0
            )
            mask = mask * logits_mask

            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

            non_zero_mask = mask.sum(1) != 0
            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask[non_zero_mask] * log_prob[non_zero_mask]).sum(1) / mask[non_zero_mask].sum(1)

            # loss
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean()
            t_loss.append(loss)

        return sum(t_loss) / len(t_loss)

    def get_loss_cate(self, source, target, s_label, t_label):  # 只对两端都出现的类别（交集 indices）计算子域内 MMD
        loss = torch.Tensor([0]).cuda()
        s_sca_label = s_label.cpu().data.numpy()
        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        indices = list(set(s_sca_label) & set(t_sca_label))
        for index in indices:
            si_batch_size = (s_sca_label == index).sum().item()
            ti_batch_size = (t_sca_label == index).sum().item()
            si = source[s_sca_label == index]
            ti = target[t_sca_label == index]
            kernels = self.guassian_kernel(si, ti, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                           fix_sigma=self.fix_sigma)
            if torch.sum(torch.isnan(sum(kernels))):
                continue
            kernels /= (si_batch_size + ti_batch_size) ** 2
            SS = kernels[:si_batch_size, :si_batch_size]
            TT = kernels[si_batch_size:, si_batch_size:]
            ST = kernels[:si_batch_size, si_batch_size:]
            TS = kernels[si_batch_size:, :si_batch_size]
            # 2023-03-06把这个地方的数字修改了一下
            loss += torch.sum(TT) - torch.sum(ST) - torch.sum(TS)
            # loss += torch.sum(SS) + torch.sum(TT) - torch.sum(ST) - torch.sum(TS)
            # loss += (torch.sum(SS) + torch.sum(TT) - torch.sum(ST) - torch.sum(TS))/(si_batch_size+ti_batch_size)
            # loss += (torch.sum(SS)/(si_batch_size**2) + torch.sum(TT)/(ti_batch_size**2) - torch.sum(ST)/(si_batch_size*ti_batch_size) - torch.sum(TS)/(si_batch_size*ti_batch_size))/(si_batch_size+ti_batch_size)

        loss /= len(indices)
        return loss

    def get_loss(self, source, target, s_label, t_label):  # 与get_loss_cate接近，但默认对目标域所有伪标签类别（不需源端出现）进行MMD
        loss = torch.Tensor([0]).cuda()
        s_sca_label = s_label.cpu().data.numpy()
        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        # indices = list(set(s_sca_label) & set(t_sca_label))
        indices = list(set(t_sca_label))
        for index in indices:
            si_batch_size = (s_sca_label == index).sum().item()
            ti_batch_size = (t_sca_label == index).sum().item()
            si = source[s_sca_label == index]
            ti = target[t_sca_label == index]
            kernels = self.guassian_kernel(si, ti,
                                           kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                           fix_sigma=self.fix_sigma)
            if torch.sum(torch.isnan(sum(kernels))):
                continue
            kernels /= (si_batch_size + ti_batch_size) ** 2
            SS = kernels[:si_batch_size, :si_batch_size]
            TT = kernels[si_batch_size:, si_batch_size:]
            ST = kernels[:si_batch_size, si_batch_size:]
            TS = kernels[si_batch_size:, :si_batch_size]
            # 2023-03-06把这个地方的数字修改了一下
            loss += torch.sum(SS) + torch.sum(TT) - torch.sum(ST) - torch.sum(TS)
            # loss += (torch.sum(SS) + torch.sum(TT) - torch.sum(ST) - torch.sum(TS))/(si_batch_size+ti_batch_size)
            # loss += (torch.sum(SS)/(si_batch_size**2) + torch.sum(TT)/(ti_batch_size**2) - torch.sum(ST)/(si_batch_size*ti_batch_size) - torch.sum(TS)/(si_batch_size*ti_batch_size))/(si_batch_size+ti_batch_size)

        loss /= len(indices)
        return loss

    def cal_weight(self, s_label, t_label):  # 为 get_loss_w 中的加权项提供基础矩阵
        source_batch_size = s_label.size()[0]
        target_batch_size = t_label.size()[0]
        s_sca_label = s_label.cpu().data.numpy()
        s_vec_label = np.eye(self.class_num)[s_sca_label]
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, self.class_num)
        s_sum[s_sum == 0] = 100
        s_vec_label = s_vec_label / s_sum

        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        # t_vec_label = np.eye(self.class_num)[t_sca_label]
        t_vec_label = t_label.cpu().data.numpy()
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, self.class_num)
        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum

        index = list(set(s_sca_label) & set(t_sca_label))
        s_mask_arr = np.zeros((source_batch_size, self.class_num))
        s_mask_arr[:, index] = 1
        s_vec_label = s_vec_label * s_mask_arr

        t_mask_arr = np.zeros((target_batch_size, self.class_num))
        t_mask_arr[:, index] = 1
        t_vec_label = t_vec_label * t_mask_arr

        weight_ss = np.matmul(s_vec_label, s_vec_label.T)
        weight_tt = np.matmul(t_vec_label, t_vec_label.T)
        weight_st = np.matmul(s_vec_label, t_vec_label.T)
        weight_ts = np.matmul(t_vec_label, s_vec_label.T)

        length = len(index)
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
            weight_ts = weight_ts / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
            weight_ts = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32'), weight_ts.astype(
            "float32")

    def get_loss_w(self, source, target, s_label, t_label, DEVICE):
        # 动态地根据每个类别在源／目标内的分布概率（soft label），给不同样本对分配不同权重，使得样本稀疏或置信高的类别对齐更强
        # 1) 过滤掉没有真实源标签的样本（label==-1）
        source = source[s_label != -1]
        s_label = s_label[s_label != -1]
        loss = torch.Tensor([0]).to(DEVICE)
        # 构建源 one‑hot 矩阵并按类别归一化
        s_sca_label = s_label.cpu().data.numpy()
        s_vec_label = np.eye(self.class_num)[s_sca_label]  # (Ns, C)
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, self.class_num)
        s_sum[s_sum == 0] = 100
        s_vec_label = s_vec_label / s_sum

        # 3) 目标标签本身是“软标签”（概率分布），直接按列归一化
        target = target[t_label != -1]
        t_label = t_label[t_label != -1]
        # 构建目标 one‑hot 矩阵并按类别归一化
        t_sca_label = t_label.cpu().data.numpy()
        t_vec_label = np.eye(self.class_num)[t_sca_label]  # (Ns, C)
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, self.class_num)
        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum

        # 4) 只对源/目标都出现过的类别做对齐
        indices = list(set(s_sca_label) & set(t_sca_label))

        for index in indices:
            # 筛出本类别在目标和源中的样本   分离出每个子域（类别）对应的源特征 si 和目标特征 ti
            target_mask = t_sca_label == index
            ti_batch_size = target_mask.sum().item()
            ti = target[target_mask]
            source_mask = s_sca_label == index
            si_batch_size = source_mask.sum().item()
            si = source[source_mask]
            # 5) 根据源 / 目标每个样本对该类别的归一化概率，计算四个权重矩阵
            weight_ss = np.matmul(s_vec_label[source_mask], s_vec_label[source_mask].T).astype('float32')
            weight_tt = np.matmul(t_vec_label[target_mask], t_vec_label[target_mask].T).astype('float32')
            weight_st = np.matmul(s_vec_label[source_mask], t_vec_label[target_mask].T).astype('float32')
            weight_ts = np.matmul(t_vec_label[target_mask], s_vec_label[source_mask].T).astype('float32')

            # # 转回 Tensor 并移到 GPU
            weight_ss = torch.from_numpy(weight_ss).to(DEVICE)
            weight_tt = torch.from_numpy(weight_tt).to(DEVICE)
            weight_st = torch.from_numpy(weight_st).to(DEVICE)
            weight_ts = torch.from_numpy(weight_ts).to(DEVICE)

            # 6) 计算该子域的多尺度 RBF 核矩阵
            kernels = self.guassian_kernel(si, ti, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                           fix_sigma=self.fix_sigma)
            if torch.sum(torch.isnan(sum(kernels))):
                continue
            # (B1,B1),(B2,B2),(B1,B2),(B2,B1)
            # 在高维（RKHS）空间中度量同子域内所有样本之间的相似度，再把结果按源-源、目-目、源-目、目-源切分
            SS = kernels[:si_batch_size, :si_batch_size].to(DEVICE)
            TT = kernels[si_batch_size:, si_batch_size:].to(DEVICE)
            ST = kernels[:si_batch_size, si_batch_size:].to(DEVICE)
            TS = kernels[si_batch_size:, :si_batch_size].to(DEVICE)
            # 7) 加权 MMD：加权后的 SS + TT – ST – TS
            loss += torch.sum(weight_ss * SS) + torch.sum(weight_tt * TT) - torch.sum(weight_st * ST) - torch.sum(
                weight_ts * TS)
        # 把各子域的加权 MMD 损失做平均，得到整体的 get_loss_w 输出
        loss /= len(indices)
        return loss

    def get_loss_lmmd(self, source, target, s_label, t_label):  # 与 get_loss_w 类似，另一种加权 MMD 实现

        source = source[s_label != -1]
        s_label = s_label[s_label != -1]

        batch_size = source.size()[0]
        weight_ss, weight_tt, weight_st, weight_ts = self.cal_weight(s_label, t_label)
        weight_ss = torch.from_numpy(weight_ss).cuda()
        weight_tt = torch.from_numpy(weight_tt).cuda()
        weight_st = torch.from_numpy(weight_st).cuda()
        weight_ts = torch.from_numpy(weight_ts).cuda()

        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                       fix_sigma=self.fix_sigma)
        loss = torch.Tensor([0]).cuda()
        if torch.sum(torch.isnan(sum(kernels))):
            return loss
        SS = kernels[:batch_size, :batch_size]
        TT = kernels[batch_size:, batch_size:]
        ST = kernels[:batch_size, batch_size:]
        TS = kernels[batch_size:, :batch_size]

        loss += torch.sum(weight_ss * SS) + torch.sum(weight_tt * TT) - torch.sum(weight_st * ST) - torch.sum(
            weight_ts * TS)
        return loss


# wd距离用于全局对齐（源域与目标域）
class SinkhornDistance(nn.Module):
    """
    Sinkhorn 距离计算模块（熵正则化最优传输）
    输入：
        - eps:     正则化系数（熵项的权重）
        - max_iter: 最大迭代次数
        - reduction: 损失降维方式（'mean'或'sum'）
    """

    def __init__(self, eps=0.1, max_iter=100, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        """
        输入：
            x: 源域特征 [batch_size, n_features]
            y: 目标域特征 [batch_size, n_features]
        返回：
            loss: Sinkhorn距离损失（标量）
            P:    传输计划矩阵（用于可视化）
        """
        batch_size, n = x.size(0), x.size(1)
        m = y.size(1)

        # 计算成对距离矩阵（成本矩阵C）
        C = self._pairwise_distances(x, y)  # [batch_size, n, m]

        # 初始化传输计划矩阵（均匀分布）
        P = torch.ones(batch_size, n, m, device=x.device) / (n * m)

        # Sinkhorn迭代
        for _ in range(self.max_iter):
            # 行归一化
            # 行归一化
            scale_n = torch.tensor(1 / n, dtype=torch.float32, device=P.device)  # 转为张量并指定设备
            P = P * scale_n / (P.sum(dim=2, keepdim=True) + 1e-8)

            # 列归一化
            scale_m = torch.tensor(1 / m, dtype=torch.float32, device=P.device)  # 转为张量并指定设备
            P = P * scale_m / (P.sum(dim=1, keepdim=True) + 1e-8)

        # 计算正则化后的Wasserstein距离
        W = (P * C).sum(dim=(1, 2))  # 按批次计算

        # 根据reduction聚合损失
        if self.reduction == 'mean':
            loss = W.mean()
        elif self.reduction == 'sum':
            loss = W.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")

        return loss, P, None  # 保持与论文中调用的接口一致

    def _pairwise_distances(self, x, y):
        """
        兼容二维输入 [batch_size, n_features]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, D] → [B, 1, D]
            y = y.unsqueeze(1)  # [B, D] → [B, 1, D]

        x_norm = (x ** 2).sum(dim=2, keepdim=True)  # [B, n, 1]
        y_norm = (y ** 2).sum(dim=2, keepdim=True)  # [B, m, 1]
        C = x_norm + y_norm.permute(0, 2, 1) - 2 * torch.bmm(x, y.permute(0, 2, 1))
        return torch.clamp(C, min=0)


# MMD算法进行全局对齐
class MMD_loss(nn.Module):    # 最大均值差异
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
                del XX, YY, XY, YX
            torch.cuda.empty_cache()
            return loss


def coral_loss(Fs: torch.Tensor, Ft: torch.Tensor) -> torch.Tensor:
    """
    计算 CORAL 损失，输入源域和目标域的特征矩阵。
    Args:
        Fs: Tensor of shape (N, d) -- 源域特征（已由 IRBlock3d+SECA 提取并池化）
        Ft: Tensor of shape (N, d) -- 目标域特征

    Returns:
        loss (Tensor): 标量， CORAL 损失
    """
    # 1. 计算均值并中心化
    Ns, d = Fs.shape
    Nt, _ = Ft.shape
    # 源域中心化
    mu_s = Fs.mean(dim=0, keepdim=True)          # (1, d)
    Fs_centered = Fs - mu_s                       # (N, d)
    # 目标域中心化
    mu_t = Ft.mean(dim=0, keepdim=True)          # (1, d)
    Ft_centered = Ft - mu_t                       # (N, d)

    # 2. 计算协方差矩阵 （无偏估计：除以 N-1）
    Cs = (Fs_centered.t() @ Fs_centered) / (Ns - 1)  # (d, d)
    Ct = (Ft_centered.t() @ Ft_centered) / (Nt - 1)  # (d, d)

    # 3. 计算 Frobenius 范数平方
    diff = Cs - Ct                                  # (d, d)
    loss = torch.sum(diff * diff)                   # scalar

    # 4. 归一化系数 1/(4 d^2)
    loss = loss / (4.0 * d * d)
    return loss


# 自适应沃瑟斯坦距离域适配（AWD-DA）
class LFCN(nn.Module):    # 生成自适应代价矩阵（显示代价矩阵）
    """
    Lightweight Feature Correlation Network to learn adaptive cost matrix C_A.
    Input:
        F_s: Tensor of shape (N, d) - source domain hint features (support+query)
        F_t: Tensor of shape (N, d) - target domain hint features (support+query)
    Output:
        C_A: Tensor of shape (N, N) - non-negative cost matrix
    """
    def __init__(self, input_dim, pooled_dim=None):
        super(LFCN, self).__init__()
        # If pooled_dim is None, keep same as input_dim
        self.pooled_dim = input_dim if pooled_dim is None else pooled_dim
        # Adaptive pooling: we can use a 1D conv or simple linear; here, use a linear layer to project to pooled_dim
        self.pool_s = nn.Linear(input_dim, self.pooled_dim)
        self.pool_t = nn.Linear(input_dim, self.pooled_dim)
        # BatchNorm2d for cost matrix? We use BatchNorm on the resulting matrix.
        self.bn = nn.BatchNorm2d(1)  # apply on single-channel matrix
        # ReLU activation
        self.relu = nn.ReLU(inplace=True)

    def forward(self, F_s, F_t):
        # F_s, F_t: (N, d)
        # Pool to (N, pooled_dim)
        A_s = self.pool_s(F_s)  # (N, pooled_dim)
        A_t = self.pool_t(F_t)  # (N, pooled_dim)
        # Compute initial cost matrix via inner product
        # C_init[i,j] = <A_s[i], A_t[j]>
        # Equivalent to A_s @ A_t^T
        C_init = torch.matmul(A_s, A_t.t())  # (N, N)
        # Add channel dimension for BatchNorm2d: shape (B=1, C=1, H=N, W=N)
        C_init_bn = C_init.unsqueeze(0).unsqueeze(0)
        C_bn = self.bn(C_init_bn)  # batchnorm on (1,1,N,N)
        C_bn = C_bn.squeeze(0)     # shape (1, N, N)
        C_bn = C_bn.squeeze(0)     # shape (N, N)
        # ReLU to ensure non-negativity
        C_A = self.relu(C_bn)
        return C_A


def sinkhorn(C, epsilon=0.1, max_iters=50, tol=1e-6):   # 利用熵正则化计算sinkhorn最优传输
    """
    Compute Sinkhorn optimal transport with entropy regularization.
    Inputs:
        C: cost matrix, Tensor of shape (N, N), non-negative
        epsilon: entropy regularization coefficient
        max_iters: maximum number of Sinkhorn iterations
        tol: convergence tolerance for u
    Returns:
        T: optimal transport matrix, shape (N, N)  最优传输矩阵
        sinkhorn_loss: torch scalar, equal to <T, C>
    """
    # Ensure C is float tensor
    N = C.size(0)
    # Uniform marginals p and q (N,)
    p = torch.full((N,), 1.0 / N, device=C.device, dtype=C.dtype)
    q = torch.full((N,), 1.0 / N, device=C.device, dtype=C.dtype)

    # Compute kernel K = exp(-C/epsilon)
    K = torch.exp(-C / epsilon)  # (N, N)
    # Initialize u and v
    u = torch.ones(N, device=C.device, dtype=C.dtype) / N
    v = torch.ones(N, device=C.device, dtype=C.dtype) / N

    # Sinkhorn iterations
    for _ in range(max_iters):
        u_prev = u
        K_v = torch.mv(K, v)  # (N,)
        u = p / (K_v + 1e-16)
        Kt_u = torch.mv(K.t(), u)
        v = q / (Kt_u + 1e-16)
        if torch.norm(u - u_prev, p=1) < tol:
            break
    # Compute transport matrix T = diag(u) K diag(v)
    T = u.unsqueeze(1) * K * v.unsqueeze(0)  # (N, N)
    # Sinkhorn loss = <T, C>
    sinkhorn_loss = torch.sum(T * C)
    return T, sinkhorn_loss


class AWD_DA_Loss(nn.Module):   # 计算Wasserstein距离使用自适应矩阵和Sinkhorn
    """
    Module to compute Wasserstein distance loss using LFCN and Sinkhorn.
    Input:
        H_s: Tensor (N, d) - source hint features
        H_t: Tensor (N, d) - target hint features
    Output:
        loss: scalar tensor
        transport_matrix: Tensor (N, N)
    """
    def __init__(self, epsilon=0.1, max_iters=50, tol=1e-6):
        super(AWD_DA_Loss, self).__init__()
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.tol = tol

    def forward(self, C_A):
        # Compute adaptive cost matrix
        # Compute Sinkhorn transport and loss
        T, sinkhorn_loss = sinkhorn(C_A, epsilon=self.epsilon,
                                     max_iters=self.max_iters,
                                     tol=self.tol)
        return sinkhorn_loss, T


class ConTeXLoss(nn.Module):
    def __init__(self, temperature=0.1, lambda_weight=1): # Context-enriched Contrastive Loss   ConTeX
        super(ConTeXLoss, self).__init__()
        self.temperature = temperature
        self.lambda_weight = lambda_weight

    def forward(self, features, labels, sample_ids):
        """
        features: Tensor of shape [batch_size, dim], 已归一化或未归一化但将归一化
        labels: Tensor of shape [batch_size], int labels
        sample_ids: Tensor of shape [batch_size], 标识原始样本 ID，不同 view 对应相同 ID
        返回: 标量损失
        """
        device = features.device
        batch_size, feat_dim = features.shape
        # 归一化特征
        features = F.normalize(features, dim=1)
        # 相似度矩阵 [batch_size, batch_size]
        logits = torch.matmul(features, features.T) / self.temperature

        labels = labels.view(-1)
        sample_ids = sample_ids.view(-1)

        # 构造 mask
        # mask_self_positive: same sample_id, exclude self
        mask_self_positive = torch.eq(sample_ids.unsqueeze(1), sample_ids.unsqueeze(0)).to(device)  # [B, B]
        mask_self_positive.fill_diagonal_(False)
        # mask_context_positive: same label, but different sample_id
        mask_same_label = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).to(device)
        mask_context_positive = mask_same_label & (~torch.eq(sample_ids.unsqueeze(1), sample_ids.unsqueeze(0)))
        # mask_context_negative: different label
        mask_context_negative = ~mask_same_label
        # mask_self_negative: different sample_id
        mask_self_negative = ~torch.eq(sample_ids.unsqueeze(1), sample_ids.unsqueeze(0))

        loss_a = 0.0
        loss_b = 0.0
        valid_a = 0
        valid_b = 0

        # 为了加速，可向量化实现，但为可读性演示使用循环计算
        # 建议在实际大 batch 中用向量操作
        for i in range(batch_size):
            # Context-based 部分 L_a(i)
            pos_idx = mask_context_positive[i]  # Tensor bool [B]
            neg_idx = mask_context_negative[i]
            if pos_idx.sum() > 0 and neg_idx.sum() > 0:
                # 对每个 context positive p 计算 log-softmax
                logits_i = logits[i]  # [B]
                # 分母: 只考虑 context negatives
                denom = torch.logsumexp(logits_i[neg_idx], dim=0)
                # 每个 p: logits_i[p] - denom
                pos_logits = logits_i[pos_idx]
                la_i = - (pos_logits - denom).mean()
                loss_a += la_i
                valid_a += 1
            # Self-based 部分 L_b(i)
            # 假设每个 sample_id 恰好有两视图，可拓展到多视图
            sp_idx = mask_self_positive[i]
            neg_idx_sb = mask_self_negative[i]
            if sp_idx.sum() > 0 and neg_idx_sb.sum() > 0:
                # 对每个 self positive（通常只有一个）
                logits_i = logits[i]
                # numerator: exp(sim(i, sp))
                sim_sp = logits_i[sp_idx]  # Tensor of sim values
                # denominator: sum exp(sim(i, n1)) over all self-negatives
                exp_neg = torch.exp(logits_i[neg_idx_sb])
                denom_sb = exp_neg.sum()
                # Lb_i = -log(1 + sum_p exp(sim_sp)/denom_sb)
                # 如果多个 self positives，可求和
                num = torch.exp(sim_sp).sum()
                lb_i = -torch.log1p(num / denom_sb)
                loss_b += lb_i
                valid_b += 1
        if valid_a > 0:
            loss_a = loss_a / valid_a
        else:
            loss_a = torch.tensor(0.0, device=device)
        if valid_b > 0:
            loss_b = loss_b / valid_b
        else:
            loss_b = torch.tensor(0.0, device=device)
        loss = self.lambda_weight * loss_a + (1 - self.lambda_weight) * loss_b
        return loss

class ConTeXLossAll(nn.Module):
    """
    Context-enriched Contrastive Loss with denominators summing over all other samples
    """
    def __init__(self, temperature=0.1, lambda_weight=0.5):
        super(ConTeXLossAll, self).__init__()
        self.temperature = temperature
        self.lambda_weight = lambda_weight

    def forward(self, features, labels, sample_ids):
        device = features.device
        batch_size, feat_dim = features.shape
        # 归一化特征
        features = F.normalize(features, dim=1)
        # 相似度矩阵 [B, B]
        logits = torch.matmul(features, features.T) / self.temperature

        labels = labels.view(-1)
        sample_ids = sample_ids.view(-1)

        # 构造 mask
        mask_self = torch.eq(sample_ids.unsqueeze(1), sample_ids.unsqueeze(0))
        mask_same_label = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))

        # Self positive: same sample_id, exclude self
        mask_self_positive = mask_self.clone()
        mask_self_positive.fill_diagonal_(False)
        # Context positive: same label, different sample_id
        mask_context_positive = mask_same_label & ~mask_self

        loss_a = 0.0
        loss_b = 0.0
        valid_a = 0
        valid_b = 0

        for i in range(batch_size):
            logits_i = logits[i]  # [B]

            # Part A: include both context positives and negatives (all except i)
            pos_ctx = mask_context_positive[i]
            all_except_i = torch.ones(batch_size, dtype=torch.bool, device=device)
            all_except_i[i] = False
            denom_a = torch.logsumexp(logits_i[all_except_i], dim=0)
            if pos_ctx.sum() > 0:
                pos_logits = logits_i[pos_ctx]
                la_i = - (pos_logits - denom_a).mean()
                loss_a += la_i
                valid_a += 1

            # Part B: include all except self as denominator
            sp = mask_self_positive[i]
            all_except_i = all_except_i  # reuse
            num = torch.exp(logits_i[sp]).sum()
            denom_b = torch.exp(logits_i[all_except_i]).sum()
            if sp.sum() > 0:
                lb_i = -torch.log1p(num / denom_b)
                loss_b += lb_i
                valid_b += 1

        loss_a = (loss_a / valid_a) if valid_a > 0 else torch.tensor(0.0, device=device)
        loss_b = (loss_b / valid_b) if valid_b > 0 else torch.tensor(0.0, device=device)
        return self.lambda_weight * loss_a + (1 - self.lambda_weight) * loss_b
