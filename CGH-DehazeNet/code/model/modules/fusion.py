from torch import nn

from .cga import SpatialAttention, ChannelAttention, PixelAttention


class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result

#########################################
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureEnhancementModule(nn.Module):
    """
    CAFR 跨谱注意力特征增强模块
    适配U‑Net跳跃融合：输入两个【通道、分辨率完全一致】的特征图
    Input:
        feat_low:  [B, C, H, W]
        feat_high: [B, C, H, W]
    Output:
        out: [B, C, H, W]
    """
    def __init__(self, channels: int):
        super().__init__()
        # 移除了原先stride=2的downsample下采样层
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_low = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.fc_high = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def _channel_cross_attention(self, low_desc: torch.Tensor, high_desc: torch.Tensor):
        B, C, _, _ = low_desc.shape
        low_vec = low_desc.reshape(B, C, 1)
        high_vec = high_desc.reshape(B, C, 1)
        weight = torch.matmul(low_vec, high_vec.transpose(1, 2))
        weight = F.softmax(weight, dim=-1)
        low_refined = torch.matmul(weight, low_vec).reshape(B, C, 1, 1)
        high_refined = torch.matmul(weight.transpose(1, 2), high_vec).reshape(B, C, 1, 1)
        return low_refined, high_refined

    def forward(self, feat_low: torch.Tensor, feat_high: torch.Tensor) -> torch.Tensor:
        low_desc = self.fc_low(self.global_pool(feat_low))
        high_desc = self.fc_high(self.global_pool(feat_high))
        low_weight, high_weight = self._channel_cross_attention(low_desc, high_desc)

        feat_low = feat_low * low_weight
        feat_high = feat_high * high_weight

        out = self.fuse(feat_low + feat_high)
        return out


