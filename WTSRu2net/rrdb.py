import torch
import torch.nn as nn
import torch.nn.functional as F
from wtconv.wtconv2d import WTConv2d
import os
import matplotlib.pyplot as plt

##############################################################################
#  Minimal Implementation of an RRDB-based SR Module
#  This is adapted from ESRGAN/Real-ESRGAN style architectures.
##############################################################################

class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block (RDB).
    Each RDB has multiple convolutions with dense connections.
    """

    def __init__(self, in_channels=128, growth_channels=32):
        super(ResidualDenseBlock, self).__init__()
        # You can add more conv layers if you want a deeper block
        # self.conv0 = WTConv2d(in_channels, in_channels, 3, 1, 1,wt_levels=2)
        self.conv1 = nn.Conv2d(in_channels, growth_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels + growth_channels, growth_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channels + 2 * growth_channels, growth_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_channels + 3 * growth_channels, in_channels, 3, 1, 1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.epoch = None
        # scale factor for residual
        self.scale_res = 0.2

    def set_epoch(self,epoch):
        self.epoch = epoch

    def forward(self, x):
        # x0 = self.relu(self.conv0(x))
        # if self.epoch is not None:
        #     self._visualize_intermediate_output(x0, "rrdb_features", f"conv0_epoch_{self.epoch}.png")
        x1 = self.relu(self.conv1(x))
        # if self.epoch is not None:
        #     self._visualize_intermediate_output(x1, "rrdb_features", f"conv0_epoch_{self.epoch}.png")
        x2 = self.relu(self.conv2(torch.cat((x, x1), dim=1)))
        if self.epoch is not None:
            self._visualize_intermediate_output(x2, "rrdb_features", f"conv0_epoch_{self.epoch}.png")
        x3 = self.relu(self.conv3(torch.cat((x, x1, x2), dim=1)))
        if self.epoch is not None:
            self._visualize_intermediate_output(x3, "rrdb_features", f"conv0_epoch_{self.epoch}.png")
        x4 = self.conv4(torch.cat((x, x1, x2, x3), dim=1))
        if self.epoch is not None:
            self._visualize_intermediate_output(x4, "rrdb_features", f"conv0_epoch_{self.epoch}.png")
        return x + self.scale_res * x4


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block (RRDB) - stacks multiple RDBs.
    """

    def __init__(self, in_channels=128, growth_channels=32, num_layers=3):
        super(RRDB, self).__init__()
        self.rdbs = nn.ModuleList(
            [ResidualDenseBlock(in_channels, growth_channels) for _ in range(num_layers)]
        )
        self.scale_res = 0.2
        self.epoch = None
    def forward(self, x):
        residual = x
        for rdb in self.rdbs:
            x = rdb(x)
        return residual + self.scale_res * x


class RRDBNet(nn.Module):
    """
    Minimal RRDB-based SR network that upsamples by factor=2.
    - in_nc: input channels (e.g. 64 for feature maps)
    - out_nc: output channels (e.g. 64 if you want to keep the dimension)
    - nf: base channel number
    - nb: number of RRDB blocks
    - scale: upsampling factor
    """

    def __init__(self, in_nc=128, out_nc=128, nf=64, nb=3, growth_channels=32, scale=2):
        super(RRDBNet, self).__init__()
        self.scale = scale

        # first convolution
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)

        # RRDB blocks
        self.rrdb_blocks = nn.ModuleList([RRDB(nf, growth_channels) for _ in range(nb)])

        # conv after RRDB
        self.conv_after_rrdb = nn.Conv2d(nf, nf, 3, 1, 1)

        # upsample x2 via PixelShuffle
        if scale == 2:
            self.upconv = nn.Conv2d(nf, nf * (2 ** 2), 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)
        else:
            raise NotImplementedError("Only scale=2 is implemented in this example.")

        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.epoch= None

    def set_epoch(self,epoch):
        self.epoch = epoch

    def forward(self, x):
        # x shape: (N, in_nc, H, W)
        fea = self.conv_first(x)

        # RRDB pipeline
        trunk = fea
        for idx, rrdb in enumerate(self.rrdb_blocks):
            trunk = rrdb(trunk)  # Pass the epoch number to visualize intermediate output
            # 可视化每一层的输出
            if self.epoch is not None:
                self._visualize_rrdb_output(trunk, "rrdb_output", f"rrdb_block{idx+1}_epoch_{self.epoch}.png")
        trunk = self.conv_after_rrdb(trunk)

        fea = fea + trunk  # long skip connection

        # upsample
        fea = self.act(self.upconv(fea))
        fea = self.pixel_shuffle(fea)
        out = self.conv_last(fea)
        # out shape: (N, out_nc, 2H, 2W)
        return out

    def _visualize_rrdb_output(self, feature_map, epoch, folder_name, file_name):
        """
        保存网络中间层的输出作为图片到指定的文件夹，文件夹会根据模块不同而命名。
        每个 epoch 都保存一次。
        """
        # 创建文件夹
        eval_folder = f"{folder_name}_epoch_{epoch}"  # 文件夹名称根据 epoch 动态创建
        os.makedirs(eval_folder, exist_ok=True)  # 如果文件夹不存在，则创建文件夹

        # 保存文件路径
        save_path = os.path.join(eval_folder, file_name)

        # 移除 batch 维度并转换形状为 [H, W, C]
        feature_map = feature_map.squeeze(0).cpu().detach()
        feature_map = feature_map.permute(1, 2, 0)  # 转换为 [H, W, C] 形状
        feature_map = feature_map.numpy()

        # 可视化并保存图像
        plt.imshow(feature_map)
        plt.axis('off')
        plt.savefig(save_path)
        plt.close()  # 关闭图像，避免内存问题

        print(f"Saved image to {save_path}")  # 打印保存路径