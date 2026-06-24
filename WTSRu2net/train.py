# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import glob
import torch
import random
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

# Import your combined U2NET + RRDB
from u2net_rrdb import U2NET_RRDB
import time
from tqdm import tqdm

###############################################################################
# 1) Segmentation Dataset
###############################################################################
class SegmentationDataset(Dataset):
    """
    Loads .jpg images (prefix 'volume-...') and their corresponding .jpg masks (prefix 'labels-...').
    Resizes both to final_size x final_size.
    Can limit the number of images via 'count'.
    """
    def __init__(self, images_dir, masks_dir, count=None, final_size=128):
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.final_size = final_size

        # 1) 只查找以 "volume-" 开头且后缀为 .jpg 的图像
        all_img_paths = sorted(glob.glob(os.path.join(self.images_dir, 'volume-*.jpg')))
        if not all_img_paths:
            raise RuntimeError(f"No .jpg files starting with 'volume-' were found in {self.images_dir}")

        # 2) 如果需要，只取前 count 张图像
        if count is not None:
            all_img_paths = all_img_paths[:count]

        self.image_paths = all_img_paths

        # 3) 定义图像和掩膜的变换（此处仅 Resize -> ToTensor，可自行添加数据增强）
        self.transform_img = transforms.Compose([
            transforms.Resize((final_size, final_size)),
            transforms.ToTensor()
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize((final_size, final_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        basename = os.path.splitext(os.path.basename(img_path))[0]

        # 4) "volume-XXX" -> "labels-XXX"
        if not basename.startswith("volume-"):
            raise ValueError(f"Image name must start with 'volume-': got {basename}")
        mask_basename = "labels" + basename[len("volume"):]
        mask_path = os.path.join(self.masks_dir, mask_basename + ".jpg")

        if not os.path.exists(mask_path):
            raise RuntimeError(f"Cannot find corresponding mask file: {mask_path}")

        # 5) 打开图像和掩膜
        img = Image.open(img_path).convert('RGB')  # 3通道
        mask = Image.open(mask_path).convert('L')  # 单通道灰度

        # print(f"Image shape: {img.size}")
        # print(f"Mask shape: {mask.size}")

        img_array = np.array(img)
        mask_array = np.array(mask)

        #调试图片输入
        # print(f"Image statistics for {img_path}:")
        # print(f"Max: {np.max(img_array)}, Min: {np.min(img_array)}, Mean: {np.mean(img_array)}")
        #
        # print(f"Mask statistics for {mask_path}:")
        # print(f"Max: {np.max(mask_array)}, Min: {np.min(mask_array)}, Mean: {np.mean(mask_array)}")
        # 7) 显示图像和掩膜

        # if idx == 0:  # 只在第一次数据读取时进行可视化
        #     plt.figure(figsize=(12, 4))
        #     plt.subplot(1, 2, 1)
        #     plt.title("Input Image")
        #     plt.imshow(img_array)
        #     plt.axis('off')
        #     plt.subplot(1, 2, 2)
        #     plt.title("Ground Truth Mask")
        #     plt.imshow(mask_array, cmap='gray')
        #     plt.axis('off')
        #     plt.show()

        # 6) 变换
        img = self.transform_img(img)
        mask = self.transform_mask(mask)

        # 7) 二值化掩膜（可根据需要决定是否保留）
        mask = (mask > 0.05).float()

        return img, mask


###############################################################################
# 2) Loss function for side outputs
###############################################################################
bce_fn = nn.BCELoss()

def segmentation_loss_sideoutputs(d0, d1, d2, d3, d4, d5, d6, mask):
    """
    Compute BCE over all 7 outputs, then sum them.
    """
    loss0 = bce_fn(d0, mask)
    loss1 = bce_fn(d1, mask)
    loss2 = bce_fn(d2, mask)
    loss3 = bce_fn(d3, mask)
    loss4 = bce_fn(d4, mask)
    loss5 = bce_fn(d5, mask)
    loss6 = bce_fn(d6, mask)
    return loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6


###############################################################################
# 3) Training function for U2NET_RRDB
###############################################################################
def train_u2net_rrdb(u2net_model, train_dataset, val_dataset=None,
                     lr=1e-4, batch_size=4, epochs=80, device='cuda'):
    """
    End-to-end training of U2NET_RRDB for segmentation.
    - train_dataset: your training data
    - val_dataset: optional, can pass None
    - lr, batch_size, epochs: hyperparameters
    - device: 'cuda' or 'cpu'
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        val_loader = None

    u2net_model = u2net_model.to(device)
    u2net_model.train()

    optimizer = optim.Adam(u2net_model.parameters(), lr=lr)
    best_val_loss = float("inf")

    #时间可视化
    total_epochs = epochs

    for epoch in range(epochs):
        # 时间可视化
        epoch_start_time = time.time()

        # u2net_model.set_epoch(epoch)#可视化self.epoch
        running_loss = 0.0


        # 使用 tqdm 进度条，显示当前 epoch 的训练进度及估计剩余时间（ETA）
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs}", unit="batch")
        for imgs, masks in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)

            # Forward
            d0, d1, d2, d3, d4, d5, d6 = u2net_model(imgs)
            loss = segmentation_loss_sideoutputs(d0, d1, d2, d3, d4, d5, d6, masks)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 可视化更新当前批次的损失显示
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_time = time.time() - epoch_start_time
        remaining_epochs = total_epochs - (epoch + 1)
        estimated_remaining_time = epoch_time * remaining_epochs
        train_loss = running_loss / len(train_loader)
        print(f"[Epoch {epoch+1}/{total_epochs}] Train Loss: {train_loss:.4f} | Epoch Time: {epoch_time:.2f}s | Estimated Remaining Time: {estimated_remaining_time/60:.2f} minutes")

        # Optional: do validation
        if val_loader is not None:
            val_loss = 0.0
            u2net_model.eval()
            with torch.no_grad():
                for vimgs, vmasks in val_loader:
                    vimgs, vmasks = vimgs.to(device), vmasks.to(device)
                    vd0, vd1, vd2, vd3, vd4, vd5, vd6 = u2net_model(vimgs)
                    vloss = segmentation_loss_sideoutputs(vd0, vd1, vd2, vd3, vd4, vd5, vd6, vmasks)
                    val_loss += vloss.item()
            val_loss /= len(val_loader)
            print(f"         Validation Loss: {val_loss:.4f}")

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(u2net_model.state_dict(), "u2net_rrdb_best.pth")
                print("         -> New best model saved.")
            u2net_model.train()

    # Final save
    torch.save(u2net_model.state_dict(), "u2net_rrdb_final.pth")
    print("Training finished. Final model saved as u2net_rrdb_final.pth")


###############################################################################
# 4) Test / Inference on 1 image
#########################test_on_one_image_######################################################
def _on_one_image_test(u2net_model, test_dataset, device='cuda'):
    """
    Runs inference on exactly 1 image from test_dataset, shows and saves the result.
    """
    if len(test_dataset) == 0:
        print("No test data available!")
        return

    img, mask = test_dataset[287]  # get the first sample
    img_t = img.unsqueeze(0).to(device)  # shape [1,3,H,W]

    u2net_model.eval()
    with torch.no_grad():
        d0, *_ = u2net_model(img_t)
    pred = d0[0, 0].cpu().numpy()  # shape (H, W)

    print(f"Input Image shape: {img.shape}")
    print(f"Ground Truth Mask shape: {mask.shape}")
    print(f"Predicted Mask shape: {pred.shape}")

    # Visualize
    plt.figure(figsize=(8,4))
    plt.subplot(1,3,1)
    plt.title("Input Image")
    plt.imshow(img.permute(1,2,0).cpu().numpy())

    plt.subplot(1,3,2)
    plt.title("Ground Truth Mask")
    plt.imshow(mask[287].cpu().numpy(), cmap='gray')

    plt.subplot(1,3,3)
    plt.title("Predicted Mask")
    plt.imshow(pred, cmap='gray')

    # 创建保存推断结果的文件夹
    eval_folder = "u2net_rrdb_output"
    os.makedirs(eval_folder, exist_ok=True)
    save_path = os.path.join(eval_folder, "test_result.png")
    plt.savefig(save_path)
    print(f"Saved evaluation figure to {save_path}")
    plt.show()



###############################################################################
# 5) 评估指标：Dice 系数
###############################################################################
def dice_coefficient(pred, target, threshold=0.5):
    """
    计算二值分割掩膜的 Dice 系数(DSC)。
    pred, target: 形状相同的 [H, W] 或 [1, H, W] (float tensor)
    threshold:    用于二值化预测
    """
    if len(pred.shape) == 3 and pred.shape[0] == 1:
        pred = pred.squeeze(0)
    if len(target.shape) == 3 and target.shape[0] == 1:
        target = target.squeeze(0)

    # 二值化预测
    pred_bin = (pred >= threshold).float()
    target_bin = (target >= threshold).float()

    intersection = torch.sum(pred_bin * target_bin)
    union = torch.sum(pred_bin) + torch.sum(target_bin)
    dice = (2.0 * intersection) / (union + 1e-7)  # 防止分母为0
    return dice.item()

def compute_global_dice(u2net_model, test_dataset, device='cuda', threshold=0.5):
    """
    计算整个测试集上的全局 Dice（DG）。
    将所有图像的交集与像素和累加后计算全局 Dice 值。
    """
    u2net_model.eval()
    total_intersection = 0.0
    total_sum = 0.0
    with torch.no_grad():
        for i in range(len(test_dataset)):
            img, mask = test_dataset[i]
            img = img.unsqueeze(0).to(device)
            mask = mask.to(device)
            d0, *_ = u2net_model(img)
            pred = d0[0, 0]  # [H,W]
            pred_bin = (pred >= threshold).float()
            target_bin = (mask[0] >= threshold).float()
            total_intersection += torch.sum(pred_bin * target_bin).item()
            total_sum += (torch.sum(pred_bin) + torch.sum(target_bin)).item()
    global_dice = 2 * total_intersection / (total_sum + 1e-7)
    return global_dice

def compute_voe(pred, target, threshold=0.5):
    """
    计算单张图像的体积重叠误差（VOE）：VOE = 1 - IoU
    pred, target: torch.Tensor, 值在0~1之间
    """
    pred_bin = (pred >= threshold).float()
    target_bin = (target >= threshold).float()
    intersection = torch.sum(pred_bin * target_bin)
    union = torch.sum(((pred_bin + target_bin) > 0).float())
    iou = intersection / (union + 1e-7)
    voe = 1 - iou
    return voe.item()

def compute_ravd(pred, target, threshold=0.5):
    """
    计算单张图像的相对绝对体积差异（RAVD）：
    RAVD = |volume(pred) - volume(target)| / volume(target)
    """
    pred_bin = (pred >= threshold).float()
    target_bin = (target >= threshold).float()
    vol_pred = torch.sum(pred_bin)
    vol_target = torch.sum(target_bin)
    ravd = torch.abs(vol_pred - vol_target) / (vol_target + 1e-7)
    return ravd.item()

def evaluate_model_metrics(u2net_model, test_dataset, device='cuda', threshold=0.5):
    """
    对整个测试集进行评估，计算每张图像的 Dice（DG）、VOE 和 RAVD，并输出平均指标。
    同时计算全局 Dice（DG）。
    """
    u2net_model.eval()
    dice_scores = []
    voe_scores = []
    ravd_scores = []
    with torch.no_grad():
        for i in range(len(test_dataset)):
            img, mask = test_dataset[i]
            img = img.unsqueeze(0).to(device)
            d0, *_ = u2net_model(img)
            pred = d0[0, 0].cpu()  # [H,W]
            target = mask[0].cpu()
            dice_score = dice_coefficient(pred, target, threshold)
            voe_score = compute_voe(pred, target, threshold)
            ravd_score = compute_ravd(pred, target, threshold)
            dice_scores.append(dice_score)
            voe_scores.append(voe_score)
            ravd_scores.append(ravd_score)
            print(f"Image {i+1}: DG (Dice) = {dice_score:.4f}, VOE = {voe_score:.4f}, RAVD = {ravd_score:.4f}")
    avg_dice = sum(dice_scores) / len(dice_scores)
    avg_voe = sum(voe_scores) / len(voe_scores)
    avg_ravd = sum(ravd_scores) / len(ravd_scores)
    global_dice = compute_global_dice(u2net_model, test_dataset, device, threshold)
    print("-------- Final Evaluation Metrics --------")
    print(f"Global Dice (DG): {global_dice:.4f}")
    print(f"Average Dice (DG): {avg_dice:.4f}")
    print(f"Average VOE: {avg_voe:.4f}")
    print(f"Average RAVD: {avg_ravd:.4f}")

###############################################################################
# 6) 可视化
###############################################################################
def visualize_all_test(u2net_model, test_dataset, device='cuda'):
    """
    对测试集中的所有图像进行推断，并将每张图像的输入、真实掩膜和预测掩膜可视化后保存。
    """
    if len(test_dataset) == 0:
        print("No test data available!")
        return

    u2net_model.eval()
    eval_folder = "u2net_rrdb_all_test_outputs"
    os.makedirs(eval_folder, exist_ok=True)

    with torch.no_grad():
        for i in range(len(test_dataset)):
            img, mask = test_dataset[i]
            img_t = img.unsqueeze(0).to(device)
            d0, *_ = u2net_model(img_t)
            pred = d0[0, 0].cpu().numpy()

            # Visualize
            plt.figure(figsize=(12,4))
            plt.subplot(1,3,1)
            plt.title("Input Image")
            plt.imshow(img.permute(1,2,0).cpu().numpy())

            plt.subplot(1,3,2)
            plt.title("Ground Truth Mask")
            plt.imshow(mask[0].cpu().numpy(), cmap='gray')

            plt.subplot(1,3,3)
            plt.title("Predicted Mask")
            plt.imshow(pred, cmap='gray')
            plt.tight_layout()

            save_path = os.path.join(eval_folder, f"test_result_{i:03d}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved visualization for test image {i} to {save_path}")

def evaluate_model_on_test(u2net_model, test_dataset, device='cuda'):
    """
    对整个测试集进行评估 (如 Dice) 并打印平均值。
    """
    if len(test_dataset) == 0:
        print("No test data available for evaluation.")
        return

    u2net_model.eval()
    dice_scores = []

    with torch.no_grad():
        for i in range(len(test_dataset)):
            img, mask = test_dataset[i]
            img = img.unsqueeze(0).to(device)      # [1,3,H,W]
            # 推断
            d0, *_ = u2net_model(img)
            pred = d0[0, 0].cpu()                 # [H,W]


            # 计算单张图像的 Dice
            dice_score = dice_coefficient(pred, mask[0])
            print(f"Image {i + 1} Dice score: {dice_score:.4f}")
            dice_scores.append(dice_score)

    mean_dice = sum(dice_scores) / len(dice_scores)
    print(f"Average Dice on Test Set: {mean_dice:.4f}")


###############################################################################
# 6) Main Function
###############################################################################
def main():
    # Paths
    train_images_dir = "train_images"
    train_masks_dir  = "train_masks"
    val_images_dir   = "test_images"
    val_masks_dir    = "test_masks"
    test_images_dir  = "val_images"
    test_masks_dir   = "val_masks"



    # Hyperparameters
    final_size  = 256
    train_count = 18156
    val_count   = 677
    test_count  = 1000
    lr          = 1e-4
    batch_size  = 12
    epochs      = 80
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Build train dataset
    train_dataset = SegmentationDataset(
        images_dir=train_images_dir,
        masks_dir=train_masks_dir,
        count=train_count,
        final_size=final_size
    )

    # Build optional val dataset
    if os.path.isdir(val_images_dir):
        val_dataset = SegmentationDataset(
            images_dir=val_images_dir,
            masks_dir=val_masks_dir,
            count=val_count,
            final_size=final_size
        )
    else:
        val_dataset = None

    # Build test dataset
    if os.path.isdir(test_images_dir):
        test_dataset = SegmentationDataset(
            images_dir=test_images_dir,
            masks_dir=test_masks_dir,
            count=test_count,
            final_size=final_size
        )
    else:
        test_dataset = None

    # Create model
    u2net_model = U2NET_RRDB(in_ch=3, out_ch=1).to(device)

    # Train
    train_u2net_rrdb(
        u2net_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,  # can be None
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        device=device
    )
    print(f"Total number of images in test dataset: {len(test_dataset)}")
    # Test / Inference
    if test_dataset:
        # 1) 可视化单张测试图像
        # _on_one_image_test(u2net_model, test_dataset, device=device)
        # 2) 整体评估 (Dice)
        visualize_all_test(u2net_model, test_dataset, device=device)
        evaluate_model_on_test(u2net_model, test_dataset, device=device)
        evaluate_model_metrics(u2net_model, test_dataset, device=device, threshold=0.5)
    else:
        print("No test folder found. Testing on the first training image instead.")
        # _on_one_image_test(u2net_model, train_dataset, device=device)


if __name__ == "__main__":
    main()
