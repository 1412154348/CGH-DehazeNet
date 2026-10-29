import torch.nn as nn
import torch.nn.functional as F
import torch

from modules import DEABlockTrain, DEBlockTrain, CGAFusion, FeatureEnhancementModule


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)



import torch
import torch.nn as nn
from torch import Tensor
class Half_Conv(nn.Module):
    def __init__(self, dim, n_div=2):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # 通道拆分：一部分做卷积，另一部分直接跳过
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x
class GCHalf_Conv(nn.Module):
    def __init__(self, dim):
        super(GCHalf_Conv, self).__init__()
        self.div_dim = int(dim / 3)
        self.remainder_dim = dim % 3

        self.p1 = Half_Conv(self.div_dim, 2)
        self.p2 = Half_Conv(self.div_dim, 2)
        self.p3 = Half_Conv(self.div_dim + self.remainder_dim, 2)

    def forward(self, x):
        y = x
        # 将输入通道切分为3组
        x1, x2, x3 = torch.split(x, [self.div_dim, self.div_dim, self.div_dim + self.remainder_dim], dim=1)
        x1 = self.p1(x1)
        x2 = self.p2(x2)
        x3 = self.p3(x3)
        x = torch.cat((x1, x2, x3), 1)
        # 残差连接
        return x + y
class CGHDehazeNet(nn.Module):
    def __init__(self, base_dim=32):
        super(CGHDehazeNet, self).__init__()
        # down‑sample 编码器
        self.down1 = nn.Sequential(nn.Conv2d(3, base_dim, kernel_size=3, stride=1, padding=1))
        self.down2 = nn.Sequential(nn.Conv2d(base_dim, base_dim*2, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        self.down3 = nn.Sequential(nn.Conv2d(base_dim*2, base_dim*4, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))

        # level 1 (最高分辨率层, base_dim)
        self.down_level1_block1 = DEBlockTrain(default_conv, base_dim, 3)
        self.down_level1_block2 = DEBlockTrain(default_conv, base_dim, 3)
        self.down_level1_block3 = DEBlockTrain(default_conv, base_dim, 3)
        self.down_level1_block4 = DEBlockTrain(default_conv, base_dim, 3)

        self.up_level1_block1 = DEBlockTrain(default_conv, base_dim, 3)
        self.up_level1_block2 = DEBlockTrain(default_conv, base_dim, 3)
        self.up_level1_block3 = DEBlockTrain(default_conv, base_dim, 3)
        self.up_level1_block4 = DEBlockTrain(default_conv, base_dim, 3)

        # level 2 (base_dim*2)
        # self.fe_level_2 = nn.Conv2d(in_channels=base_dim * 2, out_channels=base_dim * 2, kernel_size=3, stride=1, padding=1)
        self.fe_level_2 = GCHalf_Conv(base_dim * 2)
        self.down_level2_block1 = DEBlockTrain(default_conv, base_dim * 2, 3)
        self.down_level2_block2 = DEBlockTrain(default_conv, base_dim * 2, 3)
        self.down_level2_block3 = DEBlockTrain(default_conv, base_dim * 2, 3)
        self.down_level2_block4 = DEBlockTrain(default_conv, base_dim * 2, 3)

        self.up_level2_block1 = DEBlockTrain(default_conv, base_dim * 2, 3)
        self.up_level2_block2 = DEBlockTrain(default_conv, base_dim * 2, 3)
        self.up_level2_block3 = DEBlockTrain(default_conv, base_dim * 2, 3)
        self.up_level2_block4 = DEBlockTrain(default_conv, base_dim * 2, 3)

        # level 3 (最深层 base_dim*4)
        # self.fe_level_3 = nn.Conv2d(in_channels=base_dim * 4, out_channels=base_dim * 4, kernel_size=3, stride=1, padding=1)
        self.fe_level_3 = GCHalf_Conv(base_dim * 4)
        self.level3_block1 = DEABlockTrain(default_conv, base_dim * 4, 3)
        self.level3_block2 = DEABlockTrain(default_conv, base_dim * 4, 3)
        self.level3_block3 = DEABlockTrain(default_conv, base_dim * 4, 3)
        self.level3_block4 = DEABlockTrain(default_conv, base_dim * 4, 3)
        self.level3_block5 = DEABlockTrain(default_conv, base_dim * 4, 3)
        self.level3_block6 = DEABlockTrain(default_conv, base_dim * 4, 3)
        self.level3_block7 = DEABlockTrain(default_conv, base_dim * 4, 3)
        self.level3_block8 = DEABlockTrain(default_conv, base_dim * 4, 3)

        # 解码器上采样
        self.up1 = nn.Sequential(nn.ConvTranspose2d(base_dim*4, base_dim*2, kernel_size=3, stride=2, padding=1, output_padding=1),
                                  nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(base_dim*2, base_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                                  nn.ReLU(True))
        self.up3 = nn.Sequential(nn.Conv2d(base_dim, 3, kernel_size=3, stride=1, padding=1))

        # CAFR融合层，通道严格匹配
        self.mix1 = FeatureEnhancementModule(base_dim * 4)
        self.mix2 = FeatureEnhancementModule(base_dim * 2)

    def forward(self, x):
        # ----------------编码器下采样路径----------------
        x_down1 = self.down1(x)
        x_down1 = self.down_level1_block1(x_down1)
        x_down1 = self.down_level1_block2(x_down1)
        x_down1 = self.down_level1_block3(x_down1)
        x_down1 = self.down_level1_block4(x_down1)

        x_down2 = self.down2(x_down1)
        x_down2_init = self.fe_level_2(x_down2)
        x_down2_init = self.down_level2_block1(x_down2_init)
        x_down2_init = self.down_level2_block2(x_down2_init)
        x_down2_init = self.down_level2_block3(x_down2_init)
        x_down2_init = self.down_level2_block4(x_down2_init)

        x_down3 = self.down3(x_down2_init)
        x_down3_init = self.fe_level_3(x_down3)
        x1 = self.level3_block1(x_down3_init)
        x2 = self.level3_block2(x1)
        x3 = self.level3_block3(x2)
        x4 = self.level3_block4(x3)
        x5 = self.level3_block5(x4)
        x6 = self.level3_block6(x5)
        x7 = self.level3_block7(x6)
        x8 = self.level3_block8(x7)

        # 瓶颈层残差相加
        x_level3_mix = x_down3 + x8

        # ----------------解码器路径----------------
        x_up1 = self.up1(x_level3_mix)
        x_up1 = self.up_level2_block1(x_up1)
        x_up1 = self.up_level2_block2(x_up1)
        x_up1 = self.up_level2_block3(x_up1)
        x_up1 = self.up_level2_block4(x_up1)

        # CAFR跳跃融合：x_down2与x_up1通道64、分辨率128×128完全一致
        x_level2_mix = self.mix2(x_down2, x_up1)

        x_up2 = self.up2(x_level2_mix)
        x_up2 = self.up_level1_block1(x_up2)
        x_up2 = self.up_level1_block2(x_up2)
        x_up2 = self.up_level1_block3(x_up2)
        x_up2 = self.up_level1_block4(x_up2)

        out = self.up3(x_up2)
        return out

from thop import profile
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CGHDehazeNet(base_dim=32).to(device)
    test_img = torch.randn(1, 3, 256, 256).to(device)

    # 计算 FLOPs、参数量
    flops, params = profile(model, inputs=(test_img,), verbose=False)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params / 1e6:.6f} M")
    print(f"可训练参数量: {trainable_params / 1e6:.6f} M")
    print(f"计算量 FLOPs: {flops/1e9:.6f} G")
    print(f"总参数量: {params / 1e6:.6f} M")