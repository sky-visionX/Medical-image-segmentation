#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch: generate mid-labels via optical-flow-based warping (Scheme C) and
stack into one NIfTI (uint8) per Meningioma-SEG-CLASS-xxx.

当前假设目录结构为：

<root>/
  Meningioma-SEG-CLASS-001/
    sequences/
      case_00001/
        im1.png im2.png im3.png   # 这三个文件本身就是 label
      case_00002/
        im1.png im2.png im3.png
      ...
  Meningioma-SEG-CLASS-002/
    sequences/
      case_00001/
        im1.png im2.png im3.png
      ...

Output (per class):
<root>/Meningioma-SEG-CLASS-001_label.nii.gz   (uint8, labels)
<root>/Meningioma-SEG-CLASS-002_label.nii.gz   (uint8, labels)
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import nibabel as nib
from tqdm import tqdm


# ---------------- Utils ----------------

def read_gray(p: Path):
    """Read image as gray (for flow estimation)."""
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {p}")
    return img


def read_label_uint8(p: Path):
    """
    读取 label（这里就是 im1/2/3.png），统一为 uint8，支持灰度或 RGB。
    """
    lbl = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if lbl is None:
        raise FileNotFoundError(f"Failed to read label: {p}")
    if lbl.ndim == 3:
        lbl = lbl[..., 0]
    return lbl.astype(np.uint8)


def save_nii_uint8(volume_uint8: np.ndarray, out_path: Path, affine=None):
    """
    Save label volume as uint8 NIfTI.
    volume_uint8: H x W x D, dtype uint8, no normalization.
    """
    if affine is None:
        affine = np.eye(4, dtype=np.float32)
    img = nib.Nifti1Image(volume_uint8.astype(np.uint8), affine)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(out_path))


def find_three_frames(case_dir: Path, pattern="im*.png"):
    """
    在每个 case 目录下返回三帧 (im1, im2, im3)。
    优先用 im1.png, im2.png, im3.png，否则就按 pattern 找前 3 个。
    """
    p1 = case_dir / "im1.png"
    p2 = case_dir / "im2.png"
    p3 = case_dir / "im3.png"
    if p1.exists() and p2.exists() and p3.exists():
        return [p1, p2, p3]
    cand = sorted(case_dir.glob(pattern))
    if len(cand) < 3:
        raise FileNotFoundError(f"Less than 3 frames found in {case_dir}")
    return cand[:3]


def estimate_flow_farneback(img0: np.ndarray, img1: np.ndarray) -> np.ndarray:
    """
    Estimate optical flow 0->1 using OpenCV Farneback.
    Input: HxW uint8 gray
    Output: HxWx2 float32 (fx, fy)
    """
    flow = cv2.calcOpticalFlowFarneback(
        img0, img1, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.1, flags=0
    )
    return flow.astype(np.float32)


def warp_label_nearest(label: np.ndarray,
                       flow: np.ndarray,
                       t: float) -> np.ndarray:
    """
    最近邻插值 + backward warping 的 label 变形。

    forward flow 0->1:
        flow[y, x] = (fx, fy)

    对时刻 t (0~1)，近似：
        label_t(x, y) = label0(x - t * fx, y - t * fy)
    """
    assert label.ndim == 2, "Expected single-channel label image."

    h, w = label.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    src_x = grid_x - t * flow[..., 0]
    src_y = grid_y - t * flow[..., 1]

    src_x = src_x.astype(np.float32)
    src_y = src_y.astype(np.float32)

    warped = cv2.remap(
        label, src_x, src_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0  # background
    )
    return warped.astype(np.uint8)


def fuse_labels(label_from_0: np.ndarray,
                label_from_1: np.ndarray) -> np.ndarray:
    """
    两个方向 warp 出来的 label 做个简单融合：
    - 两边都 0: 0
    - 只有一边非 0: 取非 0 的那边
    - 两边都非 0: 默认保留 label_from_0
    """
    assert label_from_0.shape == label_from_1.shape

    l0 = label_from_0
    l1 = label_from_1

    out = l0.copy()
    mask0 = (l0 != 0)
    mask1 = (l1 != 0)

    out[~mask0 & mask1] = l1[~mask0 & mask1]
    return out.astype(np.uint8)


# -------------- Main per-class pipeline --------------

def process_one_class_labels(class_dir: Path,
                             flow_method: str = "farneback",
                             t_mid: float = 0.5) -> Path:
    """
    对于一个类目录 Meningioma-SEG-CLASS-001:

    - 遍历 sequences/case_*
    - 每个 case:
        * 读取 (im1, im2, im3)，它们本身就是 label
        * 用 im1, im2, im3 做灰度图算光流
        * 在 (1,2) 和 (2,3) 之间 warp 出中间标签 (t = t_mid)
        * 顺序： [lbl1, mid12, lbl2, mid23, lbl3]
    - 所有 case 串起来，stack 成 H x W x D 的 uint8 体
    - 保存到 <root>/Meningioma-SEG-CLASS-001_label.nii.gz
    """
    seq_root = class_dir / "sequences"
    if not seq_root.exists():
        raise FileNotFoundError(f"'sequences' not found under {class_dir}")

    case_dirs = sorted([d for d in seq_root.iterdir()
                        if d.is_dir() and d.name.startswith("case_")])
    if not case_dirs:
        raise FileNotFoundError(f"No case_* under {seq_root}")

    all_slices = []  # list of label arrays (H,W), in final z-order

    for case in tqdm(case_dirs, desc=f"[{class_dir.name}] label cases"):
        # 1) 每个 case 下面只有 im1.png, im2.png, im3.png
        im1_path, im2_path, im3_path = find_three_frames(case)

        # 2) 读灰度图（算光流）和 label（同一张图）
        im1 = read_gray(im1_path)
        im2 = read_gray(im2_path)
        im3 = read_gray(im3_path)

        lbl1 = read_label_uint8(im1_path)
        lbl2 = read_label_uint8(im2_path)
        lbl3 = read_label_uint8(im3_path)

        # 尺寸 sanity check
        h, w = im1.shape
        if lbl1.shape != (h, w) or lbl2.shape != (h, w) or lbl3.shape != (h, w):
            raise ValueError(f"Image/label size mismatch in {case}: "
                             f"img={im1.shape}, lbl1={lbl1.shape}, "
                             f"lbl2={lbl2.shape}, lbl3={lbl3.shape}")

        # 3) 估计光流
        if flow_method == "farneback":
            flow_12 = estimate_flow_farneback(im1, im2)  # 1->2
            flow_21 = estimate_flow_farneback(im2, im1)  # 2->1
            flow_23 = estimate_flow_farneback(im2, im3)  # 2->3
            flow_32 = estimate_flow_farneback(im3, im2)  # 3->2
        else:
            raise NotImplementedError(f"Flow method {flow_method} not implemented")

        # 4) 在 (1,2) 之间 warp label 到 t_mid
        lbl_mid12_from_1 = warp_label_nearest(lbl1, flow_12, t_mid)
        lbl_mid12_from_2 = warp_label_nearest(lbl2, flow_21, 1.0 - t_mid)
        lbl_mid12 = fuse_labels(lbl_mid12_from_1, lbl_mid12_from_2)

        # 5) 在 (2,3) 之间 warp label 到 t_mid
        lbl_mid23_from_2 = warp_label_nearest(lbl2, flow_23, t_mid)
        lbl_mid23_from_3 = warp_label_nearest(lbl3, flow_32, 1.0 - t_mid)
        lbl_mid23 = fuse_labels(lbl_mid23_from_2, lbl_mid23_from_3)

        # 6) 该 case 的最终顺序
        all_slices += [lbl1, lbl_mid12, lbl2, lbl_mid23, lbl3]

    # 7) 堆成体并保存 NIfTI
    if not all_slices:
        raise RuntimeError(f"No slices collected for {class_dir}")

    h0, w0 = all_slices[0].shape
    for idx, sl in enumerate(all_slices):
        if sl.shape != (h0, w0):
            raise ValueError(f"Slice shape mismatch at index {idx}: {sl.shape} vs {(h0, w0)}")

    vol = np.stack(all_slices, axis=2).astype(np.uint8)  # H x W x D

    out_nii = class_dir.parent / f"{class_dir.name}_label.nii.gz"
    save_nii_uint8(vol, out_nii)

    print(f"[OK] Saved label volume: {out_nii}  |  volume shape = {vol.shape} (H,W,D), dtype={vol.dtype}")
    return out_nii


# -------------- CLI --------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="Root folder containing Meningioma-SEG-CLASS-xxx folders")
    parser.add_argument("--flow_method", type=str, default="farneback",
                        choices=["farneback"],
                        help="Optical flow method for image pairs")
    parser.add_argument("--t_mid", type=float, default=0.5,
                        help="Temporal position of mid-label between frames (0~1)")
    parser.add_argument("--classes", type=str, nargs="*", default=None,
                        help="Optional CLASS list (e.g., Meningioma-SEG-CLASS-001)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    targets = []
    if args.classes:
        for name in args.classes:
            p = (root / name).resolve()
            if p.exists() and p.is_dir():
                targets.append(p)
            else:
                print(f"[WARN] skip missing class folder: {p}")
    else:
        # auto-detect all Meningioma-SEG-CLASS-xxx
        for d in sorted(root.iterdir()):
            if d.is_dir() and d.name.startswith("BRATS_"):
                targets.append(d)

    if not targets:
        raise RuntimeError("No CLASS folders found to process.")

    for class_dir in targets:
        try:
            process_one_class_labels(
                class_dir=class_dir,
                flow_method=args.flow_method,
                t_mid=args.t_mid
            )
        except Exception as e:
            print(f"[ERROR] {class_dir.name}: {e}")


if __name__ == "__main__":
    main()