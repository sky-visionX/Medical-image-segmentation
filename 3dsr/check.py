#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import math
import json
import argparse
import numpy as np
import cv2

try:
    import nibabel as nib
except ImportError:
    print("需要 `nibabel`：pip install nibabel", file=sys.stderr)
    raise

# ---------------------------
# 基础工具
# ---------------------------
def human(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return tuple(map(lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x, v))
    return v

def imread_any(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

def image_channels(arr):
    if arr is None:
        return None
    if arr.ndim == 2:
        return 1
    elif arr.ndim == 3:
        return arr.shape[2]
    return None

def sdi_load(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
        return arr
    else:
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if im is None:
            return None
        # 统一返回 float32，0-1
        if im.dtype != np.uint8:
            im = im.astype(np.float32)
            if im.max() > 1.0:
                im = np.clip(im, 0, 1)
        else:
            im = im.astype(np.float32) / 255.0
        return im

def resize_like(arr, target_hw):
    H, W = target_hw
    if arr.ndim == 2:
        return cv2.resize(arr, (W, H), interpolation=cv2.INTER_LINEAR)
    elif arr.ndim == 3:
        # 3D: [H,W,C] 或 [H,W,5]
        out = []
        for c in range(arr.shape[2]):
            out.append(cv2.resize(arr[..., c], (W, H), interpolation=cv2.INTER_LINEAR))
        return np.stack(out, axis=2)
    else:
        return arr

def ensure_rgb(img):
    # Meningioma_converted_preprocessed: np.uint8 (H,W) or (H,W,3)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def expand_to_5ch(arr):
    # arr: (H,W) 或 (H,W,1) → (H,W,5)，用复制占位
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 5, axis=2)
    return arr

def clip01(arr):
    return np.clip(arr.astype(np.float32), 0.0, 1.0)

# ---------------------------
# 检查逻辑
# ---------------------------
def check_nii(nii_path):
    info = {}
    img = nib.load(nii_path)
    data = img.get_fdata()
    hdr = img.header
    pixdim = hdr.get_zooms()[:len(data.shape)]
    info["path"] = nii_path
    info["shape"] = human(data.shape)
    info["dtype"] = str(data.dtype)
    info["pixdim"] = human(pixdim)
    info["min"] = float(np.nanmin(data))
    info["max"] = float(np.nanmax(data))
    return info

def check_images(img_dir, sample_n=5):
    out = {
        "dir": img_dir,
        "count": 0,
        "samples": []
    }
    paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    out["count"] = len(paths)
    for p in paths[:sample_n]:
        im = imread_any(p)
        ch = image_channels(im)
        shape = None if im is None else im.shape
        dtype = None if im is None else str(im.dtype)
        out["samples"].append({
            "file": os.path.basename(p),
            "shape": human(shape) if shape else None,
            "channels": ch,
            "dtype": dtype
        })
    # 推断目标尺寸（用于 SDI 对齐）
    if paths:
        im0 = imread_any(paths[0])
        if im0 is not None:
            out["target_hw"] = (im0.shape[0], im0.shape[1])
    return out

def check_sdi(sdi_dir, sample_n=5):
    out = {
        "dir": sdi_dir,
        "count_npy": 0,
        "count_png": 0,
        "samples": []
    }
    paths = sorted(glob.glob(os.path.join(sdi_dir, "dis_index*.npy"))) + \
            sorted(glob.glob(os.path.join(sdi_dir, "dis_index*.png")))
    out["count_npy"] = len([p for p in paths if p.endswith(".npy")])
    out["count_png"] = len([p for p in paths if p.endswith(".png")])
    for p in paths[:sample_n]:
        arr = sdi_load(p)
        if arr is None:
            out["samples"].append({
                "file": os.path.basename(p),
                "readable": False
            })
            continue
        shape = arr.shape
        dtype = str(arr.dtype)
        mn = float(np.nanmin(arr))
        mx = float(np.nanmax(arr))
        # 推断“通道”
        if arr.ndim == 2:
            ch = 1
        elif arr.ndim == 3:
            ch = arr.shape[2]
        else:
            ch = None
        out["samples"].append({
            "file": os.path.basename(p),
            "shape": human(shape),
            "channels": ch,
            "dtype": dtype,
            "min": mn, "max": mx
        })
    # 推断尺寸
    if paths:
        arr0 = sdi_load(paths[0])
        if arr0 is not None:
            if arr0.ndim == 2:
                out["sdi_hw"] = (arr0.shape[0], arr0.shape[1])
            elif arr0.ndim == 3:
                out["sdi_hw"] = (arr0.shape[0], arr0.shape[1])
    return out

def analyze_consistency(img_info, sdi_info, expect_rgb=True, expect_sdi_ch=5):
    report = []
    # 目标 H,W
    tgt = img_info.get("target_hw", None)
    if not tgt:
        report.append("⚠️ 无法推断切片尺寸（找不到 PNG）。")
        return report

    # 检查切片通道
    img_ch_set = set()
    for s in img_info["samples"]:
        ch = s.get("channels", None)
        if ch: img_ch_set.add(ch)
    if img_ch_set:
        if expect_rgb and (3 not in img_ch_set):
            report.append(f"❗ 切片是灰度（{sorted(img_ch_set)}），而模型常期望 RGB=3 通道。建议转 3 通道或在读取时 repeat。")
        else:
            report.append(f"✅ 切片通道检查：{sorted(img_ch_set)}")
    else:
        report.append("⚠️ 切片通道未知（可能读取失败）。")

    # 检查 SDI 尺寸 & 通道
    sdi_hw = sdi_info.get("sdi_hw", None)
    if sdi_hw and sdi_hw != tgt:
        report.append(f"❗ SDI 尺寸 {sdi_hw} 与切片尺寸 {tgt} 不一致，需 resize。")
    elif sdi_hw:
        report.append("✅ SDI 尺寸与切片一致。")
    else:
        report.append("⚠️ 无法推断 SDI 尺寸。")

    sdi_ch_set = set()
    for s in sdi_info["samples"]:
        ch = s.get("channels", None)
        if ch: sdi_ch_set.add(ch)
    if sdi_ch_set:
        if expect_sdi_ch not in sdi_ch_set:
            report.append(f"❗ SDI 通道={sorted(sdi_ch_set)}，而模型可能期望 {expect_sdi_ch} 通道（DI-RIFE/EMA-VFI 常见为 5）。")
        else:
            report.append("✅ SDI 通道检查通过。")
    else:
        report.append("⚠️ 无法推断 SDI 通道（可能读取失败）。")

    return report

# ---------------------------
# 修复操作（可选）
# ---------------------------
def fix_images_to_rgb(img_dir, out_dir=None, overwrite=False):
    paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if out_dir is None:
        out_dir = img_dir
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for p in paths:
        im = imread_any(p)
        if im is None:
            print("跳过无法读取：", p)
            continue
        if image_channels(im) == 3 and not overwrite and out_dir == img_dir:
            continue
        rgb = ensure_rgb(im)
        dst = os.path.join(out_dir, os.path.basename(p))
        cv2.imwrite(dst, rgb)
        n += 1
    print(f"✔ 转为 3 通道完成：{n} 张，输出目录：{out_dir}")

def fix_sdi(sdi_dir, target_hw, out_dir=None, expand5=False, do_clip01=False, overwrite=False, save_as="npy"):
    paths = sorted(glob.glob(os.path.join(sdi_dir, "dis_index*.npy"))) + \
            sorted(glob.glob(os.path.join(sdi_dir, "dis_index*.png")))
    if out_dir is None:
        out_dir = sdi_dir
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for p in paths:
        arr = sdi_load(p)
        if arr is None:
            print("跳过无法读取：", p); continue
        # 归一化
        if do_clip01:
            # 若是 png 已经 0-1；npy 可能越界，这里统一 clip
            if arr.ndim == 2:
                arr = clip01(arr)
            elif arr.ndim == 3:
                arr = clip01(arr)
        # resize 到目标
        arr = resize_like(arr, target_hw)
        # 扩到 5 通道
        if expand5:
            arr = expand_to_5ch(arr)
        # 保存
        name = os.path.splitext(os.path.basename(p))[0]
        dst = os.path.join(out_dir, name + (".npy" if save_as=="npy" else ".png"))
        if (not overwrite) and os.path.exists(dst):
            continue
        if save_as == "npy":
            # 保存 float16 以省空间
            np.save(dst, arr.astype(np.float16))
        else:
            # PNG 以 0-1 → 0-255 保存；多通道逐通道写（仅示例）
            to_save = arr
            if to_save.ndim == 2:
                png = (np.clip(to_save,0,1)*255+0.5).astype(np.uint8)
                cv2.imwrite(dst, png)
            else:
                # 分通道保存为 *_c#.png（如果你需要真的合并为多通道 PNG，请自行定义格式）
                for c in range(to_save.shape[2]):
                    dst_c = os.path.join(out_dir, f"{name}_c{c}.png")
                    png = (np.clip(to_save[...,c],0,1)*255+0.5).astype(np.uint8)
                    cv2.imwrite(dst_c, png)
        n += 1
    print(f"✔ SDI 修复完成：{n} 个，输出目录：{out_dir}")

# ---------------------------
# 主流程
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="3D NIfTI → 2D 切片 → SDI 全链路体检/修复")
    ap.add_argument("--nii", type=str, required=True, help="输入 3D NIfTI 路径（.nii 或 .nii.gz）")
    ap.add_argument("--images", type=str, required=True, help="切片 PNG 目录（例如 temp_slices 或 sequences/case_xx）")
    ap.add_argument("--sdi", type=str, required=True, help="SDI 目录（含 dis_index*.npy/.png）")
    ap.add_argument("--sample_n", type=int, default=5, help="每类最多展示多少样本")
    # 修复选项
    ap.add_argument("--make_images_rgb", action="store_true", help="将灰度切片批量转为 3 通道（BGR）")
    ap.add_argument("--images_out", type=str, default=None, help="切片转 3 通道后的输出目录（默认原地覆盖）")
    ap.add_argument("--resize_sdi_to", choices=["image","none"], default="none", help="将 SDI 尺寸对齐到切片尺寸")
    ap.add_argument("--sdi_out", type=str, default=None, help="SDI 修复输出目录（默认原地覆盖）")
    ap.add_argument("--expand_sdi_to5", action="store_true", help="将单通道 SDI 扩展为 5 通道")
    ap.add_argument("--clip01", action="store_true", help="将 SDI 数值裁到 [0,1]")
    ap.add_argument("--overwrite", action="store_true", help="允许覆盖已存在文件")
    args = ap.parse_args()

    print("== 检查 3D NIfTI ==")
    nii_info = check_nii(args.nii)
    print(json.dumps(nii_info, indent=2, ensure_ascii=False))

    print("\n== 检查 2D 切片 ==")
    img_info = check_images(args.images, sample_n=args.sample_n)
    print(json.dumps(img_info, indent=2, ensure_ascii=False))

    print("\n== 检查 SDI ==")
    sdi_info = check_sdi(args.sdi, sample_n=args.sample_n)
    print(json.dumps(sdi_info, indent=2, ensure_ascii=False))

    print("\n== 一致性分析 ==")
    report = analyze_consistency(img_info, sdi_info, expect_rgb=True, expect_sdi_ch=5)
    for line in report:
        print(line)

    # ====== 可选修复 ======
    if args.make_images_rgb:
        out_dir = args.images_out if args.images_out else args.images
        fix_images_to_rgb(args.images, out_dir=out_dir, overwrite=args.overwrite)

    if args.resize_sdi_to == "image":
        target_hw = img_info.get("target_hw", None)
        if not target_hw:
            print("⚠️ 未能获取切片目标尺寸，无法修复 SDI 尺寸。")
        else:
            out_dir = args.sdi_out if args.sdi_out else args.sdi
            fix_sdi(args.sdi, target_hw=target_hw, out_dir=out_dir,
                    expand5=args.expand_sdi_to5, do_clip01=args.clip01,
                    overwrite=args.overwrite, save_as="npy")

if __name__ == "__main__":
    main()
