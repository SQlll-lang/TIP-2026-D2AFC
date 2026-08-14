import numpy as np
import torchvision.transforms.functional as F
import torch
from torchvision import transforms

def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
    alpha = np.random.uniform(*alpha_range)
    noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
    return alpha * data + beta * noise

# batch image random mask spatial
def random_mask_batch_spatial(input_batch, mask_ratio): # input (batchsize, 150, 7, 7)
    batch_size = input_batch.shape[0]
    num_channels = input_batch.shape[1]
    patch_size = input_batch.shape[2]
    # 在 [0,1) 区间内为每个样本生成一个（1, patch_size, patch_size）形状的随机张量
    random_mask_spatial = torch.rand(batch_size, 1, patch_size, patch_size)
    # 对于每个位置，如果随机值大于 mask_ratio，则对应位置为 True（保留该位置）；否则为 False（遮罩该位置）
    random_mask_spatial = torch.where(random_mask_spatial > mask_ratio, torch.tensor(1.0), torch.tensor(0.0))
    masked_spatial = input_batch * random_mask_spatial

    return masked_spatial

def random_mask_batch_spectral(input_batch, num_bands_to_mask, seed=123): # input (batchsize, 150, 7, 7)
    batch_size = input_batch.shape[0]
    num_channels = input_batch.shape[1]
    # 随机光谱波段掩码
    bands = np.arange(0, 150)
    np.random.seed(seed)
    np.random.shuffle(bands)
    # 2. 取前 num_bands_to_mask 个作为要置零的通道索引
    bands_to_zero = bands[:num_bands_to_mask]  # 38
    # 先全部设为 1，再把要遮掉的那些索引置为 0
    band_mask_1d = torch.ones(num_channels, device=input_batch.device)  # shape: (150,)
    band_mask_1d[bands_to_zero] = 0.0                       # 例如把这 16 个位置设为 0
    # 4. 扩展成 (batch_size, 64, 1, 1)，以便广播到 (batch_size, 150, H, W)
    band_mask = band_mask_1d.view(1, num_channels, 1, 1)
    # 5. 最终在“空间掩码后的结果”基础上，再乘以“波段掩码”
    out = input_batch * band_mask

    return out


def random_mask_batch_spectral_v2(input_batch, mask_ratio_spectral):
    """
    按照“第一个函数”的思路：基于 mask_ratio_spectral（0~1 之间）
    对每个样本的每个通道独立掷一个随机数，若 <= mask_ratio_spectral 就置零。

    input_batch: Tensor，形状 (batch_size, num_channels, H, W)
    mask_ratio_spectral: [0,1] 单个浮点数，表示希望置零的通道比例
    seed: int or None，可选随机种子，若指定则保证可复现

    返回：Tensor，形状同 input_batch，但部分通道被随机置零。
    """
    batch_size, num_channels, H, W = input_batch.shape
    rnd = torch.rand(batch_size, num_channels, 1, 1, device=input_batch.device)
    mask_spectral = torch.where(rnd > mask_ratio_spectral, torch.tensor(1.0),torch.tensor(0.0))
    out = input_batch * mask_spectral

    return out


def ssl_data_maker(img):
    img1 = hsi_patch_augmentation(img)
    img2 = hsi_patch_augmentation(img)
    return img1, img2

def hsi_patch_augmentation(hsi_patches):
    """
    高光谱图像批量 patch 数据增强函数，适用于 (batchsize, C, 33, 33) 的 patch。
    输入：hsi_patches (torch.Tensor) - 形状为 (batchsize, C, 33, 33) 的高光谱 patch 批量
    输出：增强后的 patch 批量，空间尺寸保持为 (batchsize, C, 33, 33)
    """
    batchsize, C, H, W = hsi_patches.shape
    assert H == 7 and W == 7, "Patch size must be 7x7"

    # 1. 随机裁剪和缩放（对整个batch使用相同的裁剪参数）
    if np.random.rand() < 0.5:
        scale_range = (0.7, 1.0)  # 裁剪面积占原图面积的比例范围
        ratio_range = (3.0 / 4.0, 4.0 / 3.0)  # 宽高比范围
        size = (7, 7)  # 目标大小

        # 为整个batch生成一次随机裁剪参数
        scale = np.random.uniform(*scale_range)
        aspect_ratio = np.random.uniform(*ratio_range)
        area = H * W
        target_area = scale * area
        crop_h = int(np.sqrt(target_area / aspect_ratio))
        crop_w = int(np.sqrt(target_area * aspect_ratio))
        # 确保裁剪尺寸不超过原图
        crop_h = min(crop_h, H)
        crop_w = min(crop_w, W)
        # 随机选择裁剪位置
        i = np.random.randint(0, H - crop_h + 1)
        j = np.random.randint(0, W - crop_w + 1)

        # 对整个batch应用相同的裁剪
        cropped_patches = hsi_patches[:, :, i:i + crop_h, j:j + crop_w]
        # 调整大小到目标尺寸
        hsi_patches = F.resize(cropped_patches, size, interpolation=transforms.InterpolationMode.BICUBIC)
    # 5. 光谱噪声
    else:
        alpha_range = (0.9, 1.1)
        # 生成与x相同设备/dtype的随机alpha
        alpha = (alpha_range[1] - alpha_range[0]) * torch.rand(
            1, device=hsi_patches.device, dtype=hsi_patches.dtype
        ) + alpha_range[0]

        # 生成与x相同形状/设备/dtype的高斯噪声
        noise = torch.randn_like(hsi_patches)

        # 构造带类型和设备的beta系数
        beta = torch.tensor(1 / 25, device=hsi_patches.device, dtype=hsi_patches.dtype)

        return alpha * hsi_patches + beta * noise

    # 2. 随机水平翻转
    if np.random.rand() < 0.5:
        hsi_patches = torch.flip(hsi_patches, dims=[3])  # width 维度

    # 3. 随机垂直翻转
    if np.random.rand() < 0.5:
        hsi_patches = torch.flip(hsi_patches, dims=[2])  # height 维度

    # 4. 随机旋转
    if np.random.rand() < 0.5:
        angle = int(np.random.choice([90, 180, 270]))
        hsi_patches = F.rotate(hsi_patches, angle)

    return hsi_patches