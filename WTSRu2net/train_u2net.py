# -*- coding: utf-8 -*-
import os
import glob
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

from u2net import U2NETP
import time
from tqdm import tqdm

###############################################################################
# 2) Segmentation Dataset
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

        # 3) 定义图像和掩膜的变换（此处示例仅 Resize -> ToTensor，可自行添加数据增强）
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

        if not basename.startswith("volume-"):
            raise ValueError(f"Image name must start with 'volume-': got {basename}")

        # 将前缀 "volume-" 替换成 "labels-"
        mask_basename = "labels" + basename[len("volume"):]
        mask_path = os.path.join(self.masks_dir, mask_basename + ".jpg")

        if not os.path.exists(mask_path):
            raise RuntimeError(f"Cannot find corresponding mask file: {mask_path}")

        img = Image.open(img_path).convert('RGB')  # 3通道
        mask = Image.open(mask_path).convert('L')  # 单通道(灰度)

        # 变换
        img = self.transform_img(img)
        mask = self.transform_mask(mask)

        # 若需要将掩膜二值化
        mask = (mask > 0.5).float()

        return img, mask


###############################################################################
# 3) Side Output Loss
###############################################################################
bce_fn = nn.BCELoss()

def segmentation_loss_sideoutputs(d0, d1, d2, d3, d4, d5, d6, mask):
    """
    Sums BCE across all 7 outputs.
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
# 4) Training Function for U2NET
###############################################################################
def train_u2net(
    model,
    train_dataset,
    val_dataset=None,
    lr=1e-4,
    batch_size=32,
    epochs=80,
    device='cuda'
):
    """
    End-to-end training for the original U2NET model (from u2net.py).
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        val_loader = None

    model = model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')

    #时间可视化
    total_epochs = epochs

    for epoch in range(epochs):
        # 时间可视化
        epoch_start_time = time.time()

        running_loss = 0.0

        # 使用 tqdm 进度条，显示当前 epoch 的训练进度及估计剩余时间（ETA）
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs}", unit="batch")

        for imgs, masks in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)

            d0, d1, d2, d3, d4, d5, d6 = model(imgs)  # original U^2-Net side outputs
            loss = segmentation_loss_sideoutputs(d0, d1, d2, d3, d4, d5, d6, masks)

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
        print(f"[Epoch {epoch + 1}/{total_epochs}] Train Loss: {train_loss:.4f} | Epoch Time: {epoch_time:.2f}s | Estimated Remaining Time: {estimated_remaining_time / 60:.2f} minutes")
        epoch_loss = running_loss / len(train_loader)
        print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {epoch_loss:.4f}")

        # Optional: validation
        if val_loader is not None:
            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for vimgs, vmasks in val_loader:
                    vimgs, vmasks = vimgs.to(device), vmasks.to(device)
                    vd0, vd1, vd2, vd3, vd4, vd5, vd6 = model(vimgs)
                    vloss = segmentation_loss_sideoutputs(vd0, vd1, vd2, vd3, vd4, vd5, vd6, vmasks)
                    val_loss += vloss.item()
            val_loss /= len(val_loader)
            print(f"         Validation Loss: {val_loss:.4f}")

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "u2net_best.pth")
                print("         -> New best model saved as u2net_best.pth")

            model.train()

    # Final save
    torch.save(model.state_dict(), "u2net_final.pth")
    print("Training finished. Final model saved as u2net_final.pth")


###############################################################################
# 5) Testing on One Image
###############################################################################
def _on_one_image_test(model, test_dataset, device='cuda'):
    """
    Inference on 1 sample from test_dataset, and show the results.
    Also saves the figure to a folder "u2net_output".
    """
    if len(test_dataset) == 0:
        print("No test data available.")
        return

    img, mask = test_dataset[0]
    img_t = img.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        d0, *_ = model(img_t)
    pred = d0[0, 0].cpu().numpy()

    # 可视化
    plt.figure(figsize=(8,4))
    plt.subplot(1,3,1)
    plt.title("Input Image")
    plt.imshow(img.permute(1,2,0).cpu().numpy())

    plt.subplot(1,3,2)
    plt.title("Ground Truth Mask")
    plt.imshow(mask[0].cpu().numpy(), cmap='gray')

    plt.subplot(1,3,3)
    plt.title("Predicted Mask")
    plt.imshow(pred, cmap='gray')

    # 创建存放推断结果的文件夹并保存图片
    eval_folder = "u2net_output"
    os.makedirs(eval_folder, exist_ok=True)
    save_path = os.path.join(eval_folder, "test_result.png")
    plt.savefig(save_path)
    print(f"Saved evaluation figure to {save_path}")

    plt.show()


###############################################################################
# 6) 计算评估指标: Dice 系数
###############################################################################
def dice_coefficient(pred, target, threshold=0.5):
    """
    计算二值分割掩膜的 Dice 系数(DSC)。
    pred, target: 形状相同的 [H, W] 或 [1, H, W] (float tensor)
    threshold:    用于二值化预测
    """
    # 若 pred 是 3D: [1,H,W]，先 squeeze 到 2D
    if len(pred.shape) == 3 and pred.shape[0] == 1:
        pred = pred.squeeze(0)
    if len(target.shape) == 3 and target.shape[0] == 1:
        target = target.squeeze(0)

    # 二值化预测
    pred_bin = (pred >= threshold).float()
    target_bin = (target >= threshold).float()

    intersection = torch.sum(pred_bin * target_bin)
    union = torch.sum(pred_bin) + torch.sum(target_bin)

    # 防止分母为 0
    dice = (2.0 * intersection) / (union + 1e-7)
    return dice.item()

def compute_global_dice(model, test_dataset, device='cuda', threshold=0.5):
    """
    计算整个测试集上的全局 Dice（DG）。
    将所有图像的交集与像素和累加后计算全局 Dice 值。
    """
    model.eval()
    total_intersection = 0.0
    total_sum = 0.0
    with torch.no_grad():
        for i in range(len(test_dataset)):
            img, mask = test_dataset[i]
            img = img.unsqueeze(0).to(device)
            mask = mask.to(device)
            d0, *_ = model(img)
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

def evaluate_model_metrics(model, test_dataset, device='cuda', threshold=0.5):
    """
    对整个测试集进行评估，计算每张图像的 Dice（DG）、VOE 和 RAVD，并输出平均指标。
    同时计算全局 Dice（DG）。
    """
    model.eval()
    dice_scores = []
    voe_scores = []
    ravd_scores = []
    with torch.no_grad():
        for i in range(len(test_dataset)):
            img, mask = test_dataset[i]
            img = img.unsqueeze(0).to(device)
            d0, *_ = model(img)
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
    global_dice = compute_global_dice(model, test_dataset, device, threshold)
    print("-------- Final Evaluation Metrics --------")
    print(f"Global Dice (DG): {global_dice:.4f}")
    print(f"Average Dice (DG): {avg_dice:.4f}")
    print(f"Average VOE: {avg_voe:.4f}")
    print(f"Average RAVD: {avg_ravd:.4f}")
    metrics_str = (
        "-------- Final Evaluation Metrics --------\n"
        f"Global Dice (DG): {global_dice:.4f}\n"
        f"Average Dice (DG): {avg_dice:.4f}\n"
        f"Average VOE: {avg_voe:.4f}\n"
        f"Average RAVD: {avg_ravd:.4f}\n"
    )
    print(metrics_str)

    # 保存指标到文件
    with open("evaluation_metrics.txt", "w") as f:
        f.write(metrics_str)
def visualize_all_test(model, test_dataset, device='cuda'):
    """
    对测试集中的所有图像进行推断，并将每张图像的输入、真实掩膜和预测掩膜可视化后保存。
    """
    if len(test_dataset) == 0:
        print("No test data available!")
        return

    model.eval()
    eval_folder = "u2net_all_test_outputs"
    os.makedirs(eval_folder, exist_ok=True)

    with torch.no_grad():
        for i in range(len(test_dataset)):
            img, mask = test_dataset[i]
            img_t = img.unsqueeze(0).to(device)
            d0, *_ = model(img_t)
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
def evaluate_model_on_test(model, test_dataset, device='cuda'):
    """
    评估整个测试集上的指标（这里以 Dice 系数为例），打印平均 Dice。
    """
    model.eval()
    dice_scores = []

    with torch.no_grad():
        for i in range(len(test_dataset)):
            img, mask = test_dataset[i]
            img = img.unsqueeze(0).to(device)     # [1, 3, H, W]
            # 推断
            d0, *_ = model(img)
            pred = d0[0, 0].cpu()                # [H, W]

            # 计算单张图像的 Dice
            dice_score = dice_coefficient(pred, mask[0])
            dice_scores.append(dice_score)

    # 计算平均 Dice
    mean_dice = sum(dice_scores) / len(dice_scores)
    print(f"Average Dice on Test Dataset: {mean_dice:.4f}")
    return mean_dice


###############################################################################
# 7) Main
###############################################################################
def main():
    # Folders
    train_images_dir = "train_images"
    train_masks_dir  = "train_masks"

    val_images_dir   = "test_images"
    val_masks_dir    = "test_masks"

    test_images_dir  = "val_images"
    test_masks_dir   = "val_masks"

    # Hyperparameters
    final_size  = 256
    train_count = 18000
    val_count   = 677
    test_count  = 677
    lr          = 1e-4
    batch_size  = 24
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

    # Create the original U2NET model
    model = U2NETP(in_ch=3, out_ch=1)

    # Train
    train_u2net(
        model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        device=device
    )

    # Test on 1 image (可视化)
    if test_dataset:
        # _on_one_image_test(model, test_dataset, device=device)
        # 评估整个测试集的 Dice
        visualize_all_test(model, test_dataset, device=device)
        evaluate_model_on_test(model, test_dataset, device=device)
        evaluate_model_metrics(model, test_dataset, device=device, threshold=0.5)
    else:
        print("No test set found, skipping test visualization.")


if __name__ == "__main__":
    main()