import torch
import math
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class GLAIFormer(nn.Module):
    """GLAI-Former backbone (paper Fig. 3): ResidualStem + LSAC/LSEC + GSAT/SAAI/GSET/SEAI."""

    def __init__(self,
                 inp_channels=100,
                 dim=128,
                 patch_size=7,
                 depths=[1, 1, 1, 1],
                 num_heads_spa=[8, 8, 8, 8],
                 num_heads_spe=[7, 7, 7, 7],
                 dropout = 0.3,
                 mlp_ratio=2,
                 qkv_bias=True, qk_scale=None,
                 bias=False,
                 drop_path_rate=0.1,
                 ):
        super(GLAIFormer, self).__init__()

        self.num_layers = depths
        self.inp_channels = inp_channels
        self.patch_size = patch_size
        self.emb_size = dim
        self.dropout = dropout
        self.glai_stages = nn.ModuleList()
        print("network depth:", len(self.num_layers))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]   # 生成每个 block 的 DropPath 概率（用于训练时防止过拟合)
        for i_layer in range(len(self.num_layers)):    # stacked GLAI stages
            layer1 = GLAIStage(dim=dim,
                             patch_size=patch_size,
                             window_size=8,
                             depth=depths[i_layer],
                             num_head_spa=num_heads_spa[i_layer],
                             num_head_spe=num_heads_spe[i_layer],
                             mlp_ratio=mlp_ratio,
                             qkv_bias=qkv_bias, qk_scale=qk_scale,
                             drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                             bias=bias,)

            self.glai_stages.append(layer1)

        self.stem = ResidualStem(in_channel=inp_channels, out_channel=self.emb_size, same_shape=False)  # channel-align stem
        self.tail = default_conv(self.emb_size*3, self.emb_size, 1)   # Concat(Iout, Lspa, Lspe) -> fused embedding
        self.lsac = LSAC(input_channels=self.emb_size, patch_size=self.patch_size, feature_dim=self.emb_size)   # local spatial conv
        self.lsec = LSEC(input_channels=self.emb_size, patch_size=self.patch_size, feature_dim=self.emb_size)   # local spectral conv

        self.avgpool = nn.AvgPool2d((1, 49))
        self.classifier = nn.Linear(in_features=self.emb_size, out_features=14)  # 得到分类概率
        self.twist_batchnorm1d = nn.BatchNorm1d(14, affine=False)
        self.twist_softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.stem(x)   # (14, 128, 7, 7)
        x2 = self.lsac(x)   # LSAC: local spatial features  (14, 128, 7, 7)
        x1 = self.lsec(x)   # LSEC: local spectral features (14, 128, 7, 7)
        for i_layer in range(len(self.num_layers)):
            x = self.glai_stages[i_layer](x, x2, x1)
        #x = x + x1 + x2
        x = torch.cat([x, x2, x1], dim=1) # Concat(Iout, Lspa, Lspe) -> (B, 3C, H, W)
        x = self.tail(x)    # ([14, 128, 7, 7])
        # 转换维度
        x = rearrange(x, 'b c h w -> b c (h w)')
        x = self.avgpool(x)
        x = x.reshape((x.size(0), -1))

        # 得到分类概率 用于域对抗训练
        output = self.classifier(x)
        output = self.twist_softmax(self.twist_batchnorm1d(output))

        return x, output


class GLAIStage(nn.Module):     # stacked GLAI layers (GSAT + SAAI + GSET + SEAI)
    def __init__(self,
                 dim=90,
                 patch_size=7,
                 window_size=8,
                 depth=6,
                 num_head_spa=8,
                 num_head_spe=7,
                 mlp_ratio=2,
                 qkv_bias=True, qk_scale=None,
                 drop_path=0.0,
                 bias=False):
        super(GLAIStage, self).__init__()
        self.depth = depth
        self.glai_layers = nn.ModuleList()
        for i in range(self.depth):
            self.glai_layers.append(GLAILayer(dim=dim, input_resolution=[patch_size, patch_size], num_heads_spa=num_head_spa,
                   num_heads_spe=num_head_spe, window_size=window_size,
                   shift_size=0 if (i % 2 == 0) else window_size // 2,
                   mlp_ratio=mlp_ratio,
                   drop_path=drop_path[i],
                   qkv_bias=qkv_bias, qk_scale=qk_scale, bias=bias))

        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.act = nn.PReLU()

    def forward(self, x, y=None, z=None):
        out = self.glai_layers[0](x, y, z)
        for i in range(1, self.depth):
            out = self.glai_layers[i](out, y, z)
        out = self.conv(out) + x
        return out


class GLAILayer(nn.Module):
    r""" One GLAI interaction layer: GSAT (WMSA+GFFN) -> SAAI -> GSET (SEMA+GFFN) -> SEAI.
        Args:
            dim (int): Number of input channels.
            input_resolution (tuple[int]): Input resulotion.
            num_heads (int): Number of attention heads.
            window_size (int): Window size.
            shift_size (int): Shift size for SW-MSA.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
            drop (float, optional): Dropout rate. Default: 0.0
            attn_drop (float, optional): Attention dropout rate. Default: 0.0
            drop_path (float, optional): Stochastic depth rate. Default: 0.0
            act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        """

    def __init__(self, dim, input_resolution, num_heads_spa, num_heads_spe, window_size=7, shift_size=0, drop_path=0.0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., act_layer=nn.GELU, bias=False):
        super(GLAILayer, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads_spa = num_heads_spa
        self.num_heads_spe = num_heads_spe
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp1 = GFFN(in_features=dim, drop=drop)  # GSAT gated FFN
        self.mlp2 = GFFN(in_features=dim, drop=drop)  # GSET gated FFN


        self.gsat_attn = WMSA(          # GSAT: window-based spatial self-attention
            dim, window_size=to_2tuple(self.window_size), num_heads=self.num_heads_spa,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

        self.sema = SEMA(dim, self.num_heads_spe, bias)    # GSET: spectral multi-head self-attention

        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        self.seai = SEAI(dim)                 # spectral adaptive interaction
        self.saai = SAAI(dim)                 # spatial adaptive interaction


    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, y=None, z=None):
        B, C, H, W = x.shape   # B, C, H*W
        x = x.flatten(2).transpose(1, 2)   # B, H*W, C
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C


        attn_windows = self.gsat_attn(x_windows)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # Adaptive Interaction Module (AIM)
        # S-Map (before sigmoid)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp1(self.norm2(x), H, W))
        # x = x + self.drop_path(self.mlp1(self.norm2(x)))

        if y is not None:
            x = x.transpose(1, 2).view(B, C, H, W)
            y = self.norm1(y.flatten(2).transpose(1, 2).contiguous())
            y = y.transpose(1, 2).view(B, C, H, W).contiguous()
            y = self.dwconv(y)
            spatial_map = self.saai(y)
            # S-I
            x = x + torch.sigmoid(spatial_map) * x

        x = x.flatten(2).transpose(1, 2)
        shortcut2 = x
        x = self.norm1(x)

        x = x.transpose(1, 2).view(B, C, H, W)
        if z is not None:
            z = self.norm1(z.flatten(2).transpose(1, 2))
            z = z.transpose(1, 2).view(B, C, H, W)

        # x = self.sema(x, z)  # global spectral attention
        x = self.sema(x)
        x = x.flatten(2).transpose(1, 2)
        # FFN
        x = shortcut2 + self.drop_path(x)
        x = x + self.drop_path(self.mlp2(self.norm2(x), H, W))
        # x = x + self.drop_path(self.mlp2(self.norm2(x)))

        if z is not None:   # (B, C ,H, W)
            # C-Map (before sigmoid)
            x = x.transpose(1, 2).view(B, C, H, W).contiguous()
            channel_map = self.seai(z).permute(0, 2, 3, 1).contiguous().view(B, 1, C)
            # C-I
            x = x.flatten(2).transpose(1, 2)
            x = x + x * torch.sigmoid(channel_map)
            x = x.transpose(1, 2).view(B, C, H, W)
        return x


class SAAI(nn.Module):
    """Spatial Adaptive Interaction Module (paper Eq. 5-6)."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim // 18, kernel_size=1),
            nn.BatchNorm2d(dim // 18),
            nn.GELU(),
            nn.Conv2d(dim // 18, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)


class SEAI(nn.Module):
    """Spectral Adaptive Interaction Module (paper Eq. 7-8)."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class WMSA(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super(WMSA, self).__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y=None, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if y is not None:
            q = self.q(y).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SEMA(nn.Module):   # Spectral multi-head self-attention (paper GSET / SEMA)

    def __init__(self, dim, num_heads, bias):
        super(SEMA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y=None):
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        if y is not None:
            q = self.q(y)

        q = rearrange(q, 'b c h w -> b c (h w)')
        k = rearrange(k, 'b c h w -> b c (h w)')
        v = rearrange(v, 'b c h w -> b c (h w)')
        #
        q = rearrange(q, 'b c (head k) -> b head c k', head=self.num_heads)
        k = rearrange(k, 'b c (head k) -> b head c k', head=self.num_heads)
        v = rearrange(v, 'b c (head k) -> b head c k', head=self.num_heads)
        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # 负号的目的：「排斥自注意」（repulsive self‐attention），对相似通道施加更小的权重，鼓励模型从其它通道吸取信息
        # attn = (-attn).softmax(dim=-1)
        attn = (attn).softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c k -> b c (head k)', head=self.num_heads)
        out = rearrange(out, 'b c (h w)-> b c h w', h=h, w=w)
        # out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class GFFN(nn.Module):
    """ Spatial-Gate Feed-Forward Network.
    Args:
        in_features (int): Number of input channels.
        hidden_features (int | None): Number of hidden channels. Default: None
        out_features (int | None): Number of output channels. Default: None
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        drop (float): Dropout rate. Default: 0.0
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sg = SpatialGate(hidden_features//2)
        self.fc2 = nn.Linear(hidden_features//2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.sg(x, H, W)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x


class MLP_Block(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class SpatialGate(nn.Module):
    """ Spatial-Gate.
    Args:
        dim (int): Half of input channels.
    """
    def __init__(self, dim, act_layer=nn.GELU):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # self.norm_2 = nn.LayerNorm(dim)
        self.act = act_layer()

        # self.dwconv1 = nn.Conv3d(1, 1, kernel_size=(3,3,3), stride=1, dilation=1,padding=1) # 3D深度可分离卷积
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim) # DW Conv  深度可分离卷积
        self.pconv = nn.Conv2d(dim, dim, 1)   # 逐点卷积

    def forward(self, x, H, W):
        # Split
        x1, x2 = x.chunk(2, dim = -1)
        B, N, C = x.shape
        # x1 = self.dwconv1((self.norm_2(x1).transpose(1, 2).contiguous().view(B, C//2, H, W)).unsqueeze(1)).squeeze(1).flatten(2).transpose(-1, -2).contiguous()
        # x1 = self.conv(self.norm(x1).transpose(1, 2).contiguous().view(B, C//2, H, W)).flatten(2).transpose(-1, -2).contiguous()
        # x2 = self.pconv(self.norm_2(x2).transpose(1, 2).contiguous().view(B, C//2, H, W)).flatten(2).transpose(-1, -2).contiguous()
        x2 = self.pconv(self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C//2, H, W))).flatten(2).transpose(-1, -2).contiguous()
        # x2 = self.act(x2)

        return x1 * x2


## Spatial-Spectral Feed-Forward Network (MSFN)
class FeedForward_Gate(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward_Gate, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        self.fc1 = nn.Linear(dim, hidden_features*3)

        self.dwconv1 = nn.Conv3d(hidden_features, hidden_features, kernel_size=(3,3,3), stride=1, dilation=1,
                                 padding=1, groups=hidden_features, bias=bias)

        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features) # DW Conv  深度可分离卷积   针对空间
        self.pconv = nn.Conv2d(hidden_features, hidden_features, 1)   # 逐点卷积  针对光谱

        self.fc2 = nn.Linear(hidden_features, dim)

    def forward(self, x):
        x = self.fc1(x)
        x1,x2,x3 = x.chunk(3, dim=1)
        x1 = x1.unsqueeze(2)
        x1 = self.dwconv1(x1).squeeze(2)
        x2 = self.conv(x2)
        x3 = self.pconv(x3)

        x = F.gelu(x1) * (x2+x3)
        x = self.fc2(x)

        return x


class LSEC(nn.Module):
    def __init__(self, input_channels, patch_size, feature_dim):
        super(LSEC, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.inter_size = 24

        self.conv1 = nn.Conv3d(1, self.inter_size, kernel_size=(7, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0),
                               bias=True)
        self.bn1 = nn.BatchNorm3d(self.inter_size)
        self.activation1 = nn.ReLU()

        self.conv2 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(7, 1, 1), stride=(1, 1, 1), padding=(3, 0, 0), padding_mode='zeros', bias=True)
        self.bn2 = nn.BatchNorm3d(self.inter_size)
        self.activation2 = nn.ReLU()

        self.conv3 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(7, 1, 1), stride=(1, 1, 1), padding=(3, 0, 0), padding_mode='zeros', bias=True)
        self.bn3 = nn.BatchNorm3d(self.inter_size)
        self.activation3 = nn.ReLU()

        self.conv4 = nn.Conv3d(self.inter_size, self.feature_dim,
                               kernel_size=(((self.input_channels - 7 + 2 * 1) // 2 + 1), 1, 1), bias=True)
        self.bn4 = nn.BatchNorm3d(self.feature_dim)
        self.activation4 = nn.ReLU()


    def forward(self, x):
        x = x.unsqueeze(1)
        x1 = self.conv1(x)
        x1 = self.activation1(self.bn1(x1))

        # Residual layer 1
        residual = x1
        x1 = self.conv2(x1)
        x1 = self.activation2(self.bn2(x1))
        x1 = self.conv3(x1)
        x1 = residual + x1
        x1 = self.activation3(self.bn3(x1))

        # Convolution layer to combine rest
        x1 = self.conv4(x1)    # 压缩光谱维度为1
        x1 = self.activation4(self.bn4(x1))
        x1 = x1.reshape(x1.size(0), x1.size(1), x1.size(3), x1.size(4))  # 保证输出 B H W C

        return x1


class LSAC(nn.Module):
    def __init__(self, input_channels, patch_size, feature_dim):
        super(LSAC, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.inter_size = 24

        # Convolution layer for spatial information
        self.conv5 = nn.Conv3d(1, self.inter_size, kernel_size=(self.input_channels, 1, 1))
        self.bn5 = nn.BatchNorm3d(self.inter_size)
        self.activation5 = nn.ReLU()

        # Residual block 2
        self.conv8 = nn.Conv3d(1, self.inter_size, kernel_size=(1, 1, 1))

        self.conv6 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), padding_mode='zeros', bias=True)
        self.bn6 = nn.BatchNorm3d(self.inter_size)
        self.activation6 = nn.ReLU()
        self.conv7 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), padding_mode='zeros', bias=True)
        self.bn7 = nn.BatchNorm3d(self.inter_size)
        self.activation7 = nn.ReLU()

        self.conv9 = nn.Conv3d(self.inter_size, 1, kernel_size=(1, 1, 1))


    def forward(self, x):
        x = x.unsqueeze(1)

        # Residual layer 2
        x2 = self.conv8(x)
        residual = x2
        x2 = self.conv6(x2)
        x2 = self.activation6(self.bn6(x2))
        x2 = self.conv7(x2)
        x2 = residual + x2

        x2 = self.activation7(self.bn7(x2))

        x2 = self.conv9(x2)
        x2 = x2.reshape(x.size(0), x.size(2), x.size(3), x.size(4))     # B H W C  保证输出是这个维度

        return x2


class ResidualStem(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1, same_shape=True):
        super(ResidualStem, self).__init__()
        self.same_shape = same_shape
        # if not same_shape:
        #     strides = 2
        self.strides = strides
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1, bias=False),
            # nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        if not same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channel)
    def forward(self, x):
        out = self.block(x)
        if not self.same_shape:
            x = self.bn3(self.conv3(x))
            # x = self.conv3(x)
        # return F.relu(out + x)
        return F.relu(out + x)


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True, dilation=1, groups=1):
    if dilation==1:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=(kernel_size//2), bias=bias, groups=groups)
    elif dilation==2:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=2, bias=bias, dilation=dilation, groups=groups)

    else:
       padding = int((kernel_size - 1) / 2) * dilation
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           stride, padding=padding, bias=bias, dilation=dilation, groups=groups)


# 域判别器部分
class CrossTransformer(nn.Module):
    """
    Distribution-level cross-Transformer in D2A (paper Fig. 6 / Eq. 18).
    Given shallow embeddings f_s (N x D) from source domain and f_t (N x D) from target domain,
    splits each into left/right halves, and performs cross-attention:
      - Source-left (Query) attends to Target-right (Key/Value) -> f_s_d
      - Target-left (Query) attends to Source-right (Key/Value) -> f_t_d

    Outputs:
      f_s_d: (N//2, D)
      f_t_d: (N//2, D)
    """
    def __init__(self, embed_dim: int, num_heads: int = 4):
        """
        Args:
            embed_dim (int): Dimensionality D of input embeddings.
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # MultiheadAttention modules for both directions
        # batch_first=True expects input shape (batch_size, seq_len, embed_dim)
        self.attn_s_to_t = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attn_t_to_s = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # LayerNorm for residual outputs
        self.norm_s = nn.LayerNorm(embed_dim)
        self.norm_t = nn.LayerNorm(embed_dim)

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor):
        """
        Args:
            f_s (Tensor): Source-domain shallow embeddings, shape (N, D)
            f_t (Tensor): Target-domain shallow embeddings, shape (N, D)
        Returns:
            f_s_d (Tensor): Source-domain distribution-aligned features, shape (N//2, D)
            f_t_d (Tensor): Target-domain distribution-aligned features, shape (N//2, D)
        """
        N, D = f_s.shape
        assert f_t.shape == (N, D), "f_s and f_t must have the same shape"

        # Split each into left/right halves along the sequence dimension
        half = N // 2
        f_s_left = f_s[:half]    # shape (half, D)
        f_s_right = f_s[half:]   # shape (half, D)
        f_t_left = f_t[:half]    # shape (half, D)
        f_t_right = f_t[half:]   # shape (half, D)

        # Prepare for MultiheadAttention: add a batch dimension of 1
        # now shape (1, half, D)
        q_s = f_s_left.unsqueeze(0)
        kv_t = f_t_right.unsqueeze(0)

        # Source-left attends to Target-right
        # attn_out_s shape: (1, half, D)
        attn_out_s, _ = self.attn_s_to_t(query=q_s, key=kv_t, value=kv_t)
        # attn_out_s, _ = self.attn(query=q_s, key=kv_t, value=kv_t)
        # Remove batch dim and add residual
        f_s_d = self.norm_s((attn_out_s + q_s).squeeze(0))  # shape (half, D)

        # Target-left attends to Source-right
        q_t = f_t_left.unsqueeze(0)
        kv_s = f_s_right.unsqueeze(0)
        attn_out_t, _ = self.attn_t_to_s(query=q_t, key=kv_s, value=kv_s)
        #@ attn_out_t, _ = self.attn(query=q_t, key=kv_s, value=kv_s)
        f_t_d = self.norm_t((attn_out_t + q_t).squeeze(0))  # shape (half, D)

        return f_s_d, f_t_d


class DomainDiscriminator(nn.Module):

    def __init__(self, input_dim: int=128, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1)  # outputs raw logit
        )

    def forward(self, x: torch.Tensor):

        return self.net(x)


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)


def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


class DomainClassifier(nn.Module):
    def __init__(self):# torch.Size([1, 64, 7, 3, 3])
        super(DomainClassifier, self).__init__() #
        self.layer = nn.Sequential(
            nn.Linear(1024, 1024), #nn.Linear(320, 512), nn.Linear(FEATURE_DIM*CLASS_NUM, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

        )
        self.domain = nn.Linear(1024, 1) # 512

    def forward(self, x, iter_num):
        coeff = calc_coeff(iter_num, 1.0, 0.0, 10,10000.0)
        x.register_hook(grl_hook(coeff))
        x = self.layer(x)
        domain_y = self.domain(x)
        return domain_y


class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)    # 输入特征的数量（例如：特征维度 + 类别数）
        self.output_dim = output_dim    # 输出联合表示的维度
        # 为每个输入特征创建随机投影矩阵（不参与训练）
        self.random_matrix = nn.ParameterList([
            nn.Parameter(torch.randn(d, output_dim), requires_grad=False)
            for d in input_dim_list
        ])

    def forward(self, input_list):
        # 对每个输入进行随机线性投影
        return_list = [
            torch.mm(input_list[i].to(DEVICE), self.random_matrix[i].to(DEVICE))  # 矩阵乘法
            for i in range(self.input_num)
        ]
        # 初始化融合结果：第一个投影结果 / (output_dim^(1/输入数))
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0/len(return_list))
        # 逐元素相乘融合所有投影结果
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor


# 工具函数：one-hot 编码
def one_hot(labels, num_classes):
    return F.one_hot(labels, num_classes).float()


# 修改后的（使用普通的窗口注意力）
class SSMA_xiugai(nn.Module):
    r"""  Transformer Block:Spatial-Spectral Multi-head self-Attention (SSMA)
        Args:
            dim (int): Number of input channels.
            input_resolution (tuple[int]): Input resulotion.
            num_heads (int): Number of attention heads.
            window_size (int): Window size.
            shift_size (int): Shift size for SW-MSA.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
            drop (float, optional): Dropout rate. Default: 0.0
            attn_drop (float, optional): Attention dropout rate. Default: 0.0
            drop_path (float, optional): Stochastic depth rate. Default: 0.0
            act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, drop_path=0.0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., act_layer=nn.GELU, bias=False):
        super(SSMA_xiugai, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp1 = GFFN(in_features=dim, drop=drop)
        self.mlp2 = GFFN(in_features=dim, drop=drop)

        self.attn = SpaSA(          # 空间自注意力
            dim,  num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

        self.num_heads = num_heads

        self.sema = SEMA(dim, num_heads, bias)    # GSET / SEMA

        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        self.SAAI = nn.Sequential(                 # 空间交互辅助分支
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )
        self.SEAI = nn.Sequential(                  # 光谱交互辅助分支
            nn.Conv2d(dim, dim // 18, kernel_size=1),
            nn.BatchNorm2d(dim // 18),
            nn.GELU(),
            nn.Conv2d(dim // 18, 1, kernel_size=1)
        )

    def forward(self, x, y=None, z=None):
        B, C, H, W = x.shape   # B, C, H*W
        x = x.flatten(2).transpose(1, 2)   # B, H*W, C
        shortcut = x
        x = self.norm1(x)
        # 空间自注意力
        x = self.attn(x)

        # Adaptive Interaction Module (AIM)
        # S-Map (before sigmoid)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp1(self.norm2(x), H, W))

        if y is not None:
            x = x.transpose(1, 2).view(B, C, H, W)
            y = self.norm1(y.flatten(2).transpose(1, 2).contiguous())
            y = y.transpose(1, 2).view(B, C, H, W).contiguous()
            y = self.dwconv(y)
            spatial_map = self.SAAI(y)
            # S-I
            x = x + torch.sigmoid(spatial_map) * x

        x = x.flatten(2).transpose(1, 2)
        shortcut2 = x
        x = self.norm1(x)

        x = x.transpose(1, 2).view(B, C, H, W)
        if z is not None:
            z = self.norm1(z.flatten(2).transpose(1, 2))
            z = z.transpose(1, 2).view(B, C, H, W)

        x = self.sema(x, z)  # global spectral attention
        x = x.flatten(2).transpose(1, 2)
        # FFN
        x = shortcut2 + self.drop_path(x)
        x = x + self.drop_path(self.mlp2(self.norm2(x), H, W))

        if z is not None:
            # C-Map (before sigmoid)
            x = x.transpose(1, 2).view(B, C, H, W).contiguous()
            channel_map = self.SEAI(z).permute(0, 2, 3, 1).contiguous().view(B, 1, C)
            # C-I
            x = x.flatten(2).transpose(1, 2)
            x = x + x * torch.sigmoid(channel_map)
            x = x.transpose(1, 2).view(B, C, H, W)
        return x


class SpaSA(nn.Module):  # Transformer中的注意力模块

    def __init__(self, dim, heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):  # dim是每个注意力头的内部维度
        super().__init__()
        self.num_heads = heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5  # 1/sqrt(dim)  # 缩放因子

        self.to_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # Wq,Wk,Wv三个部分，故输出维度为dim * 3
        self.nn1 = nn.Linear(dim, dim)
        self.proj = nn.Dropout(proj_drop)

        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # ([B,1,C,N]),将Q,K,V分别取出
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale  # 计算Q和K的点积，然后进行缩放
        attn = dots.softmax(dim=-1)  # 得到注意力权重
        attn = self.attn_drop (attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # 将注意力权重应用于值 V，得到注意力输出
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block

        out = self.nn1(out)
        out = self.proj(out)
        return out


class STFRModule(nn.Module):
    """
    Source-Guided Target Feature Reconstruction (STFR) module for few-shot cross-domain tasks.
    feat_s: (N_s, D)
    feat_t: (N_t, D)
    """
    def __init__(self, feature_dim, num_visual_words=14, hidden_dim=256, momentum=0.5):
        super(STFRModule, self).__init__()
        self.K = num_visual_words
        self.D = feature_dim
        self.momentum = momentum

        # Initialize memory bank of visual words
        self.register_buffer('visual_words', torch.randn(self.K, self.D))
        nn.init.kaiming_normal_(self.visual_words)

        # Mappings for target reconstruction
        self.T1 = nn.Linear(self.D, hidden_dim)
        self.T2 = nn.Linear(self.D, hidden_dim)
        self.T3 = nn.Linear(self.D, self.D)

    @torch.no_grad()
    def update_memory(self, feat_s):
        """
        Update visual words using support features from source or target support.
        feat_s: (N_s, D)
        """
        # Cosine similarity of each support sample to each visual word
        sim = F.cosine_similarity(
            feat_s.unsqueeze(1),        # (N_s, 1, D)
            self.visual_words.unsqueeze(0),  # (1, K, D)
            dim=-1                       # (N_s, K)
        )
        # Assign each feature to nearest visual word
        idx = sim.argmax(dim=1)  # (N_s,)

        # Momentum update for each visual word
        for k in torch.unique(idx):
            mask = idx == k
            f_mean = feat_s[mask].mean(dim=0)
            v_k = self.visual_words[k]
            alpha = (F.cosine_similarity(f_mean.unsqueeze(0), v_k.unsqueeze(0)) + 1) / 2
            self.visual_words[k] = alpha * f_mean + (1 - alpha) * v_k

    def reconstruct(self, feat_t):
        """
        Reconstruct query features via visual words.
        feat_t: (N_t, D)
        returns reconstructed features (N_t, D) and attention weights (N_t, K)
        """
        qt = self.T1(feat_t)            # (N_t, hidden)
        kv = self.T2(self.visual_words) # (K, hidden)

        # Attention weights
        logits = torch.matmul(qt, kv.t())  # (N_t, K)
        w = F.softmax(logits, dim=1)

        vs_proj = self.T3(self.visual_words)  # (K, D)
        f_t_rec = torch.matmul(w, vs_proj)    # (N_t, D)
        return f_t_rec, w

    def forward(self, support_feat, query_feat, task_head, support_labels=None):
        """
        Integrate into few-shot episode:
        1. Update memory with support features (source and/or target support).
        2. Reconstruct query features.
        3. Compute consistency loss and task loss.

        support_feat: concatenated support features (N_s, D)
        query_feat: query features (N_q, D)
        task_head: nn.Module, classification or detection head{expects feat -> logits or bbox}
        support_labels: (N_s,) int labels for classification

        Returns:
          - total_loss: sum of feature consistency + task loss
          - dict of components
        """
        # 1. Update memory with support
        self.update_memory(support_feat)

        # 2. Reconstruct target features
        f_q_rec, attn = self.reconstruct(query_feat)

        # 3a. Feature-level consistency loss
        loss_feat = F.mse_loss(f_q_rec, query_feat)

        # 3b. Task-level loss:
        #    use reconstructed features for prediction
        logits_rec = task_head(f_q_rec)
        logits_q = task_head(query_feat)
        if support_labels is not None:
            # classification: use support labels for few-shot classification head
            # assume prototypical or linear head uses support to compute loss
            loss_task = F.cross_entropy(logits_rec, support_labels)
        else:
            # detection/regression: fallback to query predictions consistency
            loss_task = F.mse_loss(logits_rec, logits_q)

        total_loss = loss_feat + loss_task
        return total_loss, {
            'loss_feat': loss_feat,
            'loss_task': loss_task,
            'attention': attn,
            'reconstructed_feat': f_q_rec
        }
