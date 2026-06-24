#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import nibabel as nib
import numpy as np


def find_case_to_nii(root: Path, nii_name: str, pattern: str):
    """
    支持两种结构：
    A) root 下是 case 子目录：root/prostate_00/volume.nii.gz
    B) root 下直接是 nii 文件：root/prostate_00_0000.nii.gz
    返回：dict[case_id] = nii_path
    """
    mapping = {}

    # 情况B：直接在根目录下找 nii
    files = sorted(root.glob(pattern))
    if files:
        for f in files:
            case_id = f.name.replace(".nii.gz", "")
            mapping[case_id] = f
        return mapping

    # 情况A：根目录下是 case 子目录
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        f = d / nii_name
        if f.exists():
            mapping[d.name] = f

    return mapping


def load_3d(path: Path) -> nib.Nifti1Image:
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got shape={data.shape} from {path}")
    return img


def main():
    ap = argparse.ArgumentParser(
        description="Merge T2 + ADC 3D NIfTI into one 4D NIfTI (H,W,D,2) matching MSD Task05 datasetlabelprostate.json style."
    )
    ap.add_argument("--t2_root", required=True, type=str,
                    help="T2 数据根目录（可为 case 子目录结构或直接 nii 文件目录）")
    ap.add_argument("--adc_root", required=True, type=str,
                    help="ADC 数据根目录（结构需与 T2 对应）")
    ap.add_argument("--out_root", required=True, type=str,
                    help="输出根目录（会在其中生成合成后的 4D nii.gz）")
    ap.add_argument("--t2_nii_name", default="volume.nii.gz", type=str,
                    help="若 t2_root 下是 case 子目录结构，子目录内 nii 文件名（默认 volume.nii.gz）")
    ap.add_argument("--adc_nii_name", default="volume.nii.gz", type=str,
                    help="若 adc_root 下是 case 子目录结构，子目录内 nii 文件名（默认 volume.nii.gz）")
    ap.add_argument("--pattern", default="prostate_*.nii.gz", type=str,
                    help="若根目录下直接是 nii 文件，用该 pattern 查找（默认 prostate_*.nii.gz）")
    ap.add_argument("--out_images_subdir", default="", type=str,
                    help="可选：输出到 out_root 的某个子目录（例如 imagesTr）。默认直接输出到 out_root")
    args = ap.parse_args()

    t2_root = Path(args.t2_root).resolve()
    adc_root = Path(args.adc_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_dir = (out_root / args.out_images_subdir).resolve() if args.out_images_subdir else out_root

    if not t2_root.exists():
        raise FileNotFoundError(t2_root)
    if not adc_root.exists():
        raise FileNotFoundError(adc_root)

    out_dir.mkdir(parents=True, exist_ok=True)

    t2_map = find_case_to_nii(t2_root, args.t2_nii_name, args.pattern)
    adc_map = find_case_to_nii(adc_root, args.adc_nii_name, args.pattern)

    if not t2_map:
        raise RuntimeError(f"No T2 NIfTI found under {t2_root} (pattern={args.pattern}, nii_name={args.t2_nii_name})")
    if not adc_map:
        raise RuntimeError(f"No ADC NIfTI found under {adc_root} (pattern={args.pattern}, nii_name={args.adc_nii_name})")

    common_cases = sorted(set(t2_map.keys()) & set(adc_map.keys()))
    if not common_cases:
        raise RuntimeError("No matching case ids between T2 and ADC roots.")

    print(f"[INFO] T2 cases: {len(t2_map)}, ADC cases: {len(adc_map)}, matched: {len(common_cases)}")
    missing_t2 = sorted(set(adc_map.keys()) - set(t2_map.keys()))
    missing_adc = sorted(set(t2_map.keys()) - set(adc_map.keys()))
    if missing_t2:
        print(f"[WARN] Missing T2 for {len(missing_t2)} cases, e.g. {missing_t2[:5]}")
    if missing_adc:
        print(f"[WARN] Missing ADC for {len(missing_adc)} cases, e.g. {missing_adc[:5]}")

    for cid in common_cases:
        t2_path = t2_map[cid]
        adc_path = adc_map[cid]

        img_t2 = load_3d(t2_path)
        img_adc = load_3d(adc_path)

        t2 = img_t2.get_fdata(dtype=np.float32)
        adc = img_adc.get_fdata(dtype=np.float32)

        if t2.shape != adc.shape:
            raise ValueError(
                f"[SHAPE MISMATCH] {cid}\n"
                f"  T2 : {t2.shape} @ {t2_path}\n"
                f"  ADC: {adc.shape} @ {adc_path}\n"
                f"请先保证两者切片数/尺寸一致（插帧后应一致）。"
            )

        vol4d = np.stack([t2, adc], axis=-1).astype(np.float32)  # (H,W,D,2)

        # 用 T2 的 affine/header 作为输出参考
        out_affine = img_t2.affine
        out_header = img_t2.header.copy()
        out_header.set_data_dtype(np.float32)
        out_header.set_data_shape(vol4d.shape)

        out_path = out_dir / f"{cid}.nii.gz"   # 保持 case_id 命名（如 prostate_00_0000.nii.gz）
        nib.save(nib.Nifti1Image(vol4d, out_affine, header=out_header), str(out_path))
        print(f"[OK] {cid}: saved {out_path} | shape={vol4d.shape}")

    print("[DONE] All merged.")


if __name__ == "__main__":
    main()
