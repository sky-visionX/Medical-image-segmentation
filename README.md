# Medical-image-segmentation
本项目基于 PyTorch 实现了一个用于医学图像分割的训练与评估流程。
# U2NET-RRDB Medical Image Segmentation

## 1. 项目结构

```text
project/
├── train.py
├── u2net_rrdb.py
├── train_images/
│   ├── volume-xxx.jpg
│   └── ...
├── train_masks/
│   ├── labels-xxx.jpg
│   └── ...
├── test_images/
│   ├── volume-xxx.jpg
│   └── ...
├── test_masks/
│   ├── labels-xxx.jpg
│   └── ...
├── val_images/
│   ├── volume-xxx.jpg
│   └── ...
└── val_masks/
    ├── labels-xxx.jpg
    └── ...
```

其中：

* `train_images/`：训练图像目录；
* `train_masks/`：训练标签目录；
* `test_images/`、`test_masks/`：验证集目录；
* `val_images/`、`val_masks/`：测试集目录；
* `u2net_rrdb.py`：模型结构文件；
* `train.py`：训练、验证、测试与可视化主程序。

## 2. 数据命名规则

本项目要求图像和掩膜具有固定的命名对应关系：

```text
图像文件：volume-xxx.jpg
标签文件：labels-xxx.jpg
```

例如：

```text
train_images/volume-001.jpg
train_masks/labels-001.jpg
```

程序会自动根据图像文件名查找对应的标签文件。若找不到对应掩膜，程序会报错。

