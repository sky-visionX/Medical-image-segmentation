# -*- coding: utf-8 -*-
# dataset.py

import os
from torch.utils.data import Dataset
from PIL import Image

class SaliencyDataset(Dataset):
    """显著性检测数据集"""
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        super(SaliencyDataset, self).__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform

        # 1. 获取所有图像和掩膜文件（后缀只使用 .jpg，且图像名前缀 v，掩膜名前缀 l）
        self.images = [f for f in os.listdir(self.image_dir)
                       if f.lower().endswith('.jpg') and f.startswith('v')]
        self.masks = [f for f in os.listdir(self.mask_dir)
                      if f.lower().endswith('.jpg') and f.startswith('l')]

        # 创建掩膜文件的集合以便快速查找
        mask_set = set(self.masks)

        self.image_files = []
        self.mask_files = []

        # 2. 根据图像文件，匹配相应的掩膜文件
        for img_file in self.images:
            # 将图像文件名从 vxxx.jpg 变成 lxxx.jpg
            mask_file = 'l' + img_file[1:]
            # 判断该掩膜是否存在
            if mask_file in mask_set:
                self.image_files.append(img_file)
                self.mask_files.append(mask_file)
                print(f"匹配成功: {img_file} <--> {mask_file}")
            else:
                raise ValueError(f"对应的掩膜文件不存在: {mask_file} 对应图像 {img_file}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        # 3. 加载图像与掩膜
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # 如果需要灰度图，可以这样

        # 4. 执行预处理变换（如数据增强、标准化等）
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # 如果需要固定尺寸检查，可以加断言
        # 假设你需要的图像大小是 128x128
        # 通常情况下可以把这个检查换成自动 resize
        assert image.shape[1] == 128 and image.shape[2] == 128, f"图像尺寸不匹配: {image.shape}"
        assert mask.shape[1] == 128 and mask.shape[2] == 128, f"掩膜尺寸不匹配: {mask.shape}"

        return image, mask
