import os
import cv2
import numpy as np
import glob

def check_images(img_dir, max_num=5):
    print("="*40)
    print(f"[检查切片图片] {img_dir}")
    paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if not paths:
        print("⚠️ 没有找到 PNG 图片")
        return
    for path in paths[:max_num]:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print("❌ 无法读取:", path)
            continue
        if len(img.shape) == 2:
            ch = 1
        else:
            ch = img.shape[2]
        print(f"{os.path.basename(path)} → shape={img.shape}, 通道数={ch}, dtype={img.dtype}")

def check_sdi(sdi_dir, max_num=5):
    print("="*40)
    print(f"[检查 SDI NPY] {sdi_dir}")
    paths = sorted(glob.glob(os.path.join(sdi_dir, "*.npy")))
    if not paths:
        print("⚠️ 没有找到 NPY 文件")
        return
    for path in paths[:max_num]:
        try:
            arr = np.load(path)
            print(f"{os.path.basename(path)} → shape={arr.shape}, dtype={arr.dtype}, "
                  f"min={arr.min():.3f}, max={arr.max():.3f}")
        except Exception as e:
            print("❌ 读取失败:", path, e)

if __name__ == "__main__":
    # 修改成你自己的目录
    img_dir = "/InterpAny-Clearer-main/datasetraw/sequences/case_01"  # 切片 PNG 保存目录
    sdi_dir = "/InterpAny-Clearer-main/datasetraw/sequences/case_01"  # SDI .npy 保存目录

    check_images(img_dir)
    check_sdi(sdi_dir)
