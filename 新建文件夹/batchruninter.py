#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch: generate mid-frames with inference_img.py (variant=DR) and
stack into one NIfTI per Meningioma-SEG-CLASS-xxx.

Directory layout (example):
<root>/
  Meningioma-SEG-CLASS-001/
    sequences/
      case_00001/
        im1.png im2.png im3.png   # or other *.png (auto-pick first 3 sorted)
      case_00002/
        ...
  Meningioma-SEG-CLASS-002/
    sequences/...
Output:
<root>/Meningioma-SEG-CLASS-001.nii.gz
<root>/Meningioma-SEG-CLASS-002.nii.gz
"""

import os
import sys
import shlex
import glob
import argparse
from pathlib import Path

import cv2
import numpy as np
import nibabel as nib
from tqdm import tqdm


# ---------------- Utils ----------------

def normalize01(x, eps=1e-8):
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def read_gray_uint8(p):
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {p}")
    return img


def stack_pngs_to_volume(png_list):
    arrs = []
    for p in png_list:
        im = read_gray_uint8(p)
        arrs.append(im.astype(np.float32) / 255.0)
    vol = np.stack(arrs, axis=2)  # H x W x D
    return vol


def save_nii(volume_float01, out_path, affine=None):
    if affine is None:
        affine = np.eye(4, dtype=np.float32)
    img = nib.Nifti1Image(volume_float01.astype(np.float32), affine)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(out_path))


def run_inference_img(img0, img1, save_dir, model, variant, checkpoint, iters, num=1, extra_args=None):
    """
    Call inference_img.py to produce mid-frame(s) between img0 and img1.
    We will scan save_dir/*.png and pick the median frame as 'mid.png'.
    """
    img0 = str(Path(img0).resolve())
    img1 = str(Path(img1).resolve())
    save_dir = str(Path(save_dir).resolve())
    checkpoint = str(Path(checkpoint).resolve())
    os.makedirs(save_dir, exist_ok=True)

    num_str = ' '.join(map(str, num)) if isinstance(num, (list, tuple)) else str(num)
    cmd = [
        "python", "inference_img.py",
        "--img0", img0,
        "--img1", img1,
        "--model", model,
        "--variant", variant,
        "--checkpoint", checkpoint,
        "--save_dir", save_dir,
        "--num", num_str
    ]
    if iters and int(iters) > 0:
        cmd += ["--iters", str(int(iters))]

    # TODO (if you later add external SDI support):
    # if extra_args and "sdi_path" in extra_args:
    #     cmd += ["--sdi_path", str(Path(extra_args["sdi_path"]).resolve())]

    print("[INFO] Running:", " ".join(shlex.quote(c) for c in cmd))
    proc = os.popen(" ".join(shlex.quote(c) for c in cmd))
    _ = proc.read()  # collect stdout
    ret = proc.close()
    if ret is not None:
        raise RuntimeError(f"inference_img.py exited with code {ret}")

    pngs = sorted(glob.glob(os.path.join(save_dir, "*.png")))
    if not pngs:
        raise FileNotFoundError(f"No PNG generated in {save_dir}")
    mid = pngs[len(pngs)//2]
    # standardize name
    mid_std = os.path.join(save_dir, "mid.png")
    try:
        if Path(mid) != Path(mid_std):
            import shutil
            if os.path.exists(mid_std):
                os.remove(mid_std)
            shutil.copy2(mid, mid_std)
        return mid_std
    except Exception:
        return mid


def find_three_frames(case_dir, pattern="im*.png"):
    """Return three frames (P1,P2,P3). If 'im1.png..im3.png' exists use them; else pick first 3 sorted."""
    p1 = case_dir / "im1.png"
    p2 = case_dir / "im2.png"
    p3 = case_dir / "im3.png"
    if p1.exists() and p2.exists() and p3.exists():
        return [p1, p2, p3]
    cand = sorted(case_dir.glob(pattern))
    if len(cand) < 3:
        raise FileNotFoundError(f"Less than 3 frames found in {case_dir}")
    return cand[:3]


# -------------- Main pipeline --------------

def process_one_class(class_dir: Path,
                      model="RIFE",
                      variant="DR",
                      checkpoint="checkpoints/RIFE/DR-RIFE-pro",
                      iters=2,
                      num=1,
                      tmp_dirname="interp_pairs") -> Path:
    """
    For a class folder like Meningioma-SEG-CLASS-001,
    - iterate sequences/case_*,
    - generate mid(1-2), mid(2-3) via inference_img.py,
    - append order [im1, mid12, im2, mid23, im3] from each case into one big list,
    - stack and save to <root>/Meningioma-SEG-CLASS-001.nii.gz
    """
    seq_root = class_dir / "sequences"
    if not seq_root.exists():
        raise FileNotFoundError(f"'sequences' not found under {class_dir}")

    case_dirs = sorted([d for d in seq_root.iterdir() if d.is_dir() and d.name.startswith("case_")])
    if not case_dirs:
        raise FileNotFoundError(f"No case_* under {seq_root}")

    all_slices = []  # list of PNG paths in final z-order

    for case in tqdm(case_dirs, desc=f"[{class_dir.name}] cases"):
        # 1) pick three frames
        im1, im2, im3 = find_three_frames(case)

        # 2) call inference to produce two mids
        pair_root = case / tmp_dirname
        pair12 = pair_root / "pair_1_2"
        pair23 = pair_root / "pair_2_3"
        pair12.mkdir(parents=True, exist_ok=True)
        pair23.mkdir(parents=True, exist_ok=True)

        mid12 = pair12 / "mid.png"
        if not mid12.exists():
            _mid12 = run_inference_img(
                img0=im1, img1=im2, save_dir=pair12,
                model=model, variant=variant, checkpoint=checkpoint, iters=iters, num=num
            )
            # ensure standard name
            if Path(_mid12) != mid12:
                import shutil
                shutil.copy2(_mid12, mid12)

        mid23 = pair23 / "mid.png"
        if not mid23.exists():
            _mid23 = run_inference_img(
                img0=im2, img1=im3, save_dir=pair23,
                model=model, variant=variant, checkpoint=checkpoint, iters=iters, num=num
            )
            if Path(_mid23) != mid23:
                import shutil
                shutil.copy2(_mid23, mid23)

        # 3) push into final order for this case
        all_slices += [im1, mid12, im2, mid23, im3]

    # 4) stack to volume and save NIfTI
    vol = stack_pngs_to_volume([str(p) for p in all_slices])
    vol = normalize01(vol)  # 保证 [0,1]

    out_nii = class_dir.parent / f"{class_dir.name}.nii.gz"  # 保存到 <root> 下
    save_nii(vol, out_nii)

    print(f"[OK] Saved: {out_nii}  |  volume shape = {vol.shape} (H,W,D)")
    return out_nii


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="Root folder containing Meningioma-SEG-CLASS-xxx folders")
    parser.add_argument("--model", type=str, default="RIFE",
                        choices=["RIFE", "IFRNet", "AMT-S", "EMA-VFI"])
    parser.add_argument("--variant", type=str, default="DR", choices=["T", "D", "TR", "DR"])
    parser.add_argument("--checkpoint", type=str, default="checkpoints/RIFE/DR-RIFE-pro",
                        help="Path to checkpoint directory (consistent with inference_img.py)")
    parser.add_argument("--iters", type=int, default=2, help="iters for TR/DR")
    parser.add_argument("--num", type=int, nargs="+", default=[1], help="--num for inference_img.py")
    parser.add_argument("--classes", type=str, nargs="*", default=None,
                        help="Optional list to only process specific CLASS folders (basename, e.g., Meningioma-SEG-CLASS-001)")
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
            if d.is_dir() and d.name.startswith("hippocampus_"):
                targets.append(d)

    if not targets:
        raise RuntimeError("No CLASS folders found to process.")

    # absolute checkpoint to satisfy inference_img.py's path join
    args.checkpoint = str(Path(args.checkpoint).resolve())

    for class_dir in targets:
        try:
            process_one_class(
                class_dir=class_dir,
                model=args.model,
                variant=args.variant,
                checkpoint=args.checkpoint,
                iters=args.iters,
                num=args.num
            )
        except Exception as e:
            print(f"[ERROR] {class_dir.name}: {e}")


if __name__ == "__main__":
    main()
