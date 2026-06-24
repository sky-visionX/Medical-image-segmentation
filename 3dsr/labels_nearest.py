#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import nibabel as nib


def nn_resample_along_axis(vol: np.ndarray, new_len: int, axis: int, tie_break: str = "next") -> np.ndarray:
    """
    仅沿某一轴做“最近邻插帧/重采样”，保证标签为整数类别，不做任何线性插值。

    tie_break:
      - "next": 当恰好在中点(0.5)时偏向后一个切片
      - "prev": 当恰好在中点(0.5)时偏向前一个切片
    """
    old_len = vol.shape[axis]
    if new_len == old_len:
        return vol
    if old_len <= 0:
        raise ValueError("old_len <= 0")
    if new_len <= 0:
        raise ValueError("new_len <= 0")

    # 新坐标映射到旧坐标：0..old_len-1
    old_pos = np.linspace(0, old_len - 1, new_len)

    # 自定义“最近邻”索引（避免 numpy round 的 bankers rounding）
    if tie_break == "next":
        idx = np.floor(old_pos + 0.5).astype(np.int64)   # 0.5 -> 1
    elif tie_break == "prev":
        idx = np.ceil(old_pos - 0.5).astype(np.int64)    # 0.5 -> 0
    else:
        raise ValueError("tie_break must be 'next' or 'prev'")

    idx = np.clip(idx, 0, old_len - 1)
    return np.take(vol, idx, axis=axis)


def depth_from_ref_image(ref_path: Path, axis: int) -> int:
    """
    参考图像可能是 3D 或 4D(多通道)，取 axis 对应的深度长度。
    对于 (H,W,D,2)，axis=2 对应 D。
    """
    img = nib.load(str(ref_path))
    shp = img.shape
    if len(shp) == 3:
        return shp[axis]
    if len(shp) == 4:
        return shp[axis]
    raise ValueError(f"Unexpected ref image shape: {shp} ({ref_path})")


def unique_set(a: np.ndarray) -> set:
    return set(np.unique(a).tolist())


def main():
    ap = argparse.ArgumentParser(
        description="Nearest-neighbor label interpolation along slice axis (3D label stays single-channel)."
    )
    ap.add_argument("--labels_dir", required=True, type=str,
                    help="输入 labelsTr 目录（3D label .nii.gz）")
    ap.add_argument("--out_dir", required=True, type=str,
                    help="输出 labelsTr 目录（插帧后）")
    ap.add_argument("--pattern", default="prostate_*.nii.gz", type=str,
                    help="病例匹配 pattern（默认 prostate_*.nii.gz）")
    ap.add_argument("--axis", default=2, type=int,
                    help="插帧/重采样的轴（默认 2，即 H,W,D 的 D 轴）")
    ap.add_argument("--tie_break", default="next", choices=["next", "prev"],
                    help="最近邻遇到中点时偏向前/后切片（默认 next）")

    # 两种模式二选一：
    ap.add_argument("--ref_images_dir", default=None, type=str,
                    help="可选：参考图像目录（如合成后的 imagesTr）。若提供，则逐例对齐 label 深度到参考图像深度。")
    ap.add_argument("--insert", default=None, type=int,
                    help="可选：固定插帧数：每相邻两层之间插入 insert 层。比如 insert=1 -> 新深度=2*D-1。")

    # 保存 dtype
    ap.add_argument("--dtype", default=None, type=str,
                    help="可选：输出 dtype，如 uint8/int16。默认保持原 label dtype。")

    args = ap.parse_args()

    labels_dir = Path(args.labels_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not labels_dir.exists():
        raise FileNotFoundError(labels_dir)

    ref_dir = Path(args.ref_images_dir).resolve() if args.ref_images_dir else None
    if ref_dir is not None and not ref_dir.exists():
        raise FileNotFoundError(ref_dir)

    if args.ref_images_dir is None and args.insert is None:
        raise ValueError("必须提供 --ref_images_dir 或 --insert 其中之一")

    if args.insert is not None and args.insert < 0:
        raise ValueError("--insert 必须 >= 0")

    files = sorted(labels_dir.glob(args.pattern))
    if not files:
        raise RuntimeError(f"No labels matched {args.pattern} in {labels_dir}")

    print(f"[INFO] Found {len(files)} labels in {labels_dir}")

    for lbl_path in files:
        case_id = lbl_path.name.replace(".nii.gz", "")
        lbl_img = nib.load(str(lbl_path))
        lbl = np.asanyarray(lbl_img.dataobj)  # 保留整数标签

        if lbl.ndim != 3:
            raise ValueError(f"{case_id}: label must be 3D, got {lbl.shape}")

        # 目标深度
        if ref_dir is not None:
            ref_path = ref_dir / f"{case_id}.nii.gz"
            if not ref_path.exists():
                raise FileNotFoundError(f"{case_id}: missing ref image {ref_path}")
            target_len = depth_from_ref_image(ref_path, axis=args.axis)
        else:
            old_len = lbl.shape[args.axis]
            target_len = old_len + (old_len - 1) * args.insert  # insert=1 -> 2D-1

        # 最近邻插帧/重采样
        lbl_new = nn_resample_along_axis(lbl, new_len=target_len, axis=args.axis, tie_break=args.tie_break)

        # dtype 保持/指定
        if args.dtype is not None:
            out_dtype = np.dtype(args.dtype)
            lbl_new = lbl_new.astype(out_dtype, copy=False)
        else:
            lbl_new = lbl_new.astype(lbl.dtype, copy=False)

        # 类别检查：不应产生新类别
        u_before = unique_set(lbl)
        u_after = unique_set(lbl_new)
        new_vals = u_after - u_before
        if new_vals:
            print(f"[WARN] {case_id}: 出现新标签值 {sorted(new_vals)}（理论上不该发生，请检查 dtype/输入）")

        # 保存：保持原 affine/header（或你也可改成参考图像 affine）
        out_path = out_dir / f"{case_id}.nii.gz"
        header = lbl_img.header.copy()
        header.set_data_dtype(lbl_new.dtype)
        header.set_data_shape(lbl_new.shape)
        nib.save(nib.Nifti1Image(lbl_new, lbl_img.affine, header=header), str(out_path))

        print(f"[OK] {case_id}: {lbl.shape} -> {lbl_new.shape} | saved {out_path}")

    print("[DONE] All labels processed.")


if __name__ == "__main__":
    main()
