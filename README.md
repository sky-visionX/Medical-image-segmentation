# Medical-image-segmentation
本项目基于 PyTorch 实现了一个用于医学图像分割的训练与评估流程。
# WTSRu2net
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
#3dsr （医疗插帧）
## 1. 数据集结构

```text
/home/dell/wzh/new/InterpAny-Clearer-main/
└── dataset/
    ├── Meningioma-SEG-CLASS-001/
    │   ├── sequences/
    │   │   ├── case_001/
    │   │   │   ├── im1.png
    │   │   │   ├── im2.png
    │   │   │   └── im3.png
    │   │   │
    │   │   ├── case_002/
    │   │   │   ├── im1.png
    │   │   │   ├── im2.png
    │   │   │   └── im3.png
    │   │   │
    │   │   └── case_xxx/
    │   │       ├── im1.png
    │   │       ├── im2.png
    │   │       └── im3.png
    │   │
    │   └── vimeo_triplet/
    │       ├── tri_trainlist.txt
    │       └── tri_testlist.txt
    │
    ├── Meningioma-SEG-CLASS-002/
    │   ├── sequences/
    │   │   ├── case_001/
    │   │   │   ├── im1.png
    │   │   │   ├── im2.png
    │   │   │   └── im3.png
    │   │   │
    │   │   ├── case_002/
    │   │   │   ├── im1.png
    │   │   │   ├── im2.png
    │   │   │   └── im3.png
    │   │   │
    │   │   └── case_xxx/
    │   │       ├── im1.png
    │   │       ├── im2.png
    │   │       └── im3.png
    │   │
    │   └── vimeo_triplet/
    │       ├── tri_trainlist.txt
    │       └── tri_testlist.txt
    │
    └── Meningioma-SEG-CLASS-xxx/
        ├── sequences/
        │   ├── case_001/
        │   │   ├── im1.png
        │   │   ├── im2.png
        │   │   └── im3.png
        │   │
        │   └── case_xxx/
        │       ├── im1.png
        │       ├── im2.png
        │       └── im3.png
        │
        └── vimeo_triplet/
            ├── tri_trainlist.txt
            └── tri_testlist.txt
```

##2.命令
```text
制作dis_index命令
CUDA_VISIBLE_DEVICES=0,1 python multiprocess_create_dis_index.py --num_gpus 2 --num_workers 1 （--path /home/dell/wzh/new/InterpAny-Clearer-main/dataset/vimeo_triplet/ --sample_list_path tri_testlist.txt） --sample_length 3
```

```text
Test
python inference_img.py --img0 /home/dell/wzh/InterpAny-Clearer-main/volume-11-283.jpg --img1 /home/dell/wzh/InterpAny-Clearer-main/volume-11-285.jpg --model RIFE --variant DR --checkpoint /home/dell/wzh/InterpAny-Clearer-main/checkpoints/RIFE/DR-RIFE --save_dir /home/dell/wzh/InterpAny-Clearer-main --num 1  --gif
```

```text
train
CUDA_VISIBLE_DEVICES=0,1  python -m torch.distributed.launch --nproc_per_node=2 --master_port 29502 ./models/RIFE/train_sdi_m_mask.py --world_size 2 --batch_size 1 --exp_name EMA-VFI_sdi_m_triplet --use_sdi --triplet --data_path ./dataset/vimeo_triplet
```

```text
批量生成中间帧，并合成3d
python batchruninter.py --root /home/dell/wzh/new/Meningioma_converted_preprocessed --model RIFE --variant DR --checkpoint /home/dell/wzh/new/InterpAny-Clearer-main/checkpoints/RIFE/DR-RIFE-pro --iters 2 --num 1
```
