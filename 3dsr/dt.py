#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import argparse

import numpy as np
import nibabel as nib
from tqdm import tqdm

# raft mode needs these
import cv2
import torch
from easydict import EasyDict as edict


# ---------------------- Dt core (same as your dis_index.py) ----------------------
def cosine_project_ratio(v0t, v01):
    """
    v0t, v01: (H, W, 2) float32
    return Dt: (H, W) in [0,1]
    """
    inner = v0t[..., 0] * v01[..., 0] + v0t[..., 1] * v01[..., 1]
    denom = v01[..., 0] ** 2 + v01[..., 1] ** 2
    mag = np.sqrt(denom + 1e-12)

    eps = np.percentile(mag, 10)
    valid = denom > (eps ** 2 + 1e-12)

    out = np.zeros_like(denom, dtype=np.float32)
    out[valid] = (inner[valid] / (denom[valid] + 1e-12)).astype(np.float32)

    out = np.clip(out, 0.0, 1.0)
    out[~np.isfinite(out)] = 0.5

    # median fill (optional but usually stabilizes)
    med = cv2.medianBlur((out * 255).astype(np.uint8), 5).astype(np.float32) / 255.0
    out[~valid] = med[~valid]
    return np.clip(out, 0.0, 1.0)


# ---------------------- helpers ----------------------
def to_uint8_like(img2d: np.ndarray) -> np.ndarray:
    """
    Convert a 2D slice (float/int) to uint8 [0,255] robustly.
    We use percentile clipping to avoid outliers.
    """
    x = img2d.astype(np.float32)
    if not np.isfinite(x).any():
        return np.zeros_like(x, dtype=np.uint8)

    lo = np.percentile(x[np.isfinite(x)], 0.5)
    hi = np.percentile(x[np.isfinite(x)], 99.5)
    if hi <= lo + 1e-6:
        return np.zeros_like(x, dtype=np.uint8)

    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + 1e-8)
    x = (x * 255.0).astype(np.uint8)
    return x


def get_z_dim(arr: np.ndarray, z_axis: int) -> int:
    return arr.shape[z_axis]


def get_slice(arr: np.ndarray, z: int, z_axis: int) -> np.ndarray:
    """
    Return 2D slice as (H,W). We assume arr is 3D.
    """
    if z_axis == -1 or z_axis == 2:
        return arr[:, :, z]
    elif z_axis == 0:
        return arr[z, :, :]
    elif z_axis == 1:
        return arr[:, z, :]
    else:
        raise ValueError("z_axis must be -1/0/1/2")


def set_slice(vol: np.ndarray, z: int, z_axis: int, slice2d: np.ndarray):
    """
    Set 2D slice into vol (same shape as image).
    """
    if z_axis == -1 or z_axis == 2:
        vol[:, :, z] = slice2d
    elif z_axis == 0:
        vol[z, :, :] = slice2d
    elif z_axis == 1:
        vol[:, z, :] = slice2d
    else:
        raise ValueError("z_axis must be -1/0/1/2")


def infer_out_name(stem: str, as_modality: bool) -> str:
    """
    If as_modality=True and input endswith _0000, output _0001.
    Else output stem + _dt.
    """
    if as_modality and stem.endswith("_0000"):
        return stem[:-5] + "_0001"
    return stem + "_dt"


# ---------------------- RAFT wrapper ----------------------
class FlowEstimator:
    def __init__(self, checkpoint: str, raft_root: str = "", iters=20, device="cuda"):
        # optionally add RAFT repo paths
        if raft_root:
            rr = Path(raft_root)
            sys.path.append(str(rr))
            sys.path.append(str(rr / "core"))

        from raft import RAFT
        try:
            from utils.utils import InputPadder
        except ImportError:
            from RAFT.core.utils.utils import InputPadder

        self.InputPadder = InputPadder
        self.device = device
        self.iters = iters

        args = edict({'mixed_precision': False, 'small': False, 'alternate_corr': False})
        model = torch.nn.DataParallel(RAFT(args))
        state = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(state)
        self.model = model.module.to(self.device).eval()

    def _slice_to_tensor(self, slice2d_uint8: np.ndarray):
        """
        (H,W) uint8 -> (1,3,H,W) float in 0..255
        """
        if slice2d_uint8.ndim != 2:
            raise ValueError("slice must be 2D")
        img = np.repeat(slice2d_uint8[..., None], 3, axis=2)  # (H,W,3)
        ten = torch.from_numpy(img).permute(2, 0, 1).float()[None]  # (1,3,H,W)
        return ten.to(self.device)

    @torch.no_grad()
    def flow(self, s0_uint8: np.ndarray, s1_uint8: np.ndarray) -> np.ndarray:
        """
        Return flow (H,W,2) float32.
        """
        im0 = self._slice_to_tensor(s0_uint8)
        im1 = self._slice_to_tensor(s1_uint8)
        padder = self.InputPadder(im0.shape)
        im0, im1 = padder.pad(im0, im1)
        _, flow_up = self.model(im0, im1, iters=self.iters, test_mode=True)
        flow_up = padder.unpad(flow_up)  # (1,2,H,W)
        v = flow_up[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
        return v


# ---------------------- Dt generators ----------------------
def make_dt_uniform(arr_shape, z_axis: int, scope: str, insert_k: int = None) -> np.ndarray:
    """
    scope:
      - global: Dt(z)=z/(D-1)
      - local : if insert_k given, Dt repeats in each segment: (z%step)/step
    """
    dt = np.zeros(arr_shape, dtype=np.float32)
    D = arr_shape[z_axis]
    if D <= 1:
        return dt

    if scope == "global":
        for z in range(D):
            val = z / float(D - 1)
            set_slice(dt, z, z_axis, np.full(get_slice(dt, z, z_axis).shape, val, dtype=np.float32))
        return dt

    if scope == "local":
        if insert_k is None:
            raise RuntimeError("scope=local 需要 --insert K")
        step = insert_k + 1
        for z in range(D):
            val = (z % step) / float(step)
            set_slice(dt, z, z_axis, np.full(get_slice(dt, z, z_axis).shape, val, dtype=np.float32))
        return dt

    raise ValueError("scope must be global or local")


def make_dt_raft(img_vol: np.ndarray, z_axis: int, scope: str, insert_k: int, raft_est: FlowEstimator) -> np.ndarray:
    """
    Using RAFT on 2D slices from the interpolated volume.
    Assumes uniform interpolation factor K: between keyframes inserted K slices (step=K+1).
    """
    dt = np.zeros_like(img_vol, dtype=np.float32)
    D = img_vol.shape[z_axis]
    if D <= 1:
        return dt

    step = insert_k + 1
    if (D - 1) % step != 0:
        raise RuntimeError(f"当前 D={D} 与 K={insert_k} 不匹配：(D-1) 必须能被 (K+1) 整除。")

    # cache keyframe slices in uint8
    # keyframes at indices 0, step, 2*step, ...
    key_idx = list(range(0, D, step))

    # preconvert all slices to uint8 for speed
    slices_u8 = [to_uint8_like(get_slice(img_vol, z, z_axis)) for z in range(D)]

    # global denom for mapping
    denom_global = float(D - 1)

    for seg in range(len(key_idx) - 1):
        i0 = key_idx[seg]
        i1 = key_idx[seg + 1]

        # keyframes: set as constants
        if scope == "global":
            val0 = i0 / denom_global
            val1 = i1 / denom_global
        else:
            # local: keyframe start=0, end=1
            val0 = 0.0
            val1 = 1.0

        set_slice(dt, i0, z_axis, np.full(get_slice(dt, i0, z_axis).shape, val0, dtype=np.float32))
        set_slice(dt, i1, z_axis, np.full(get_slice(dt, i1, z_axis).shape, val1, dtype=np.float32))

        # V01 once
        v01 = raft_est.flow(slices_u8[i0], slices_u8[i1])

        # intermediates
        for j in range(1, step):
            it = i0 + j
            v0t = raft_est.flow(slices_u8[i0], slices_u8[it])
            dt_local = cosine_project_ratio(v0t, v01)  # (H,W) in [0,1]

            if scope == "local":
                # directly use local dt in [0,1]
                set_slice(dt, it, z_axis, dt_local.astype(np.float32))
            else:
                # map to global continuous position: (i0 + dt_local*step) / (D-1)
                dt_global = (float(i0) + dt_local * float(step)) / denom_global
                set_slice(dt, it, z_axis, dt_global.astype(np.float32))

    return dt


# ---------------------- main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagesTr", type=str, required=True, help="插帧后 imagesTr 目录（里面是 *.nii 或 *.nii.gz）")
    ap.add_argument("--out_dir", type=str, required=True, help="输出 Dt NIfTI 的目录")
    ap.add_argument("--mode", type=str, default="uniform", choices=["uniform", "raft"],
                    help="uniform: 仅按层号生成Dt；raft: RAFT+distance-index 生成Dt")
    ap.add_argument("--scope", type=str, default="global", choices=["global", "local"],
                    help="global: 全卷0->1单调；local: 每个插帧段内0->1重复（需要K）")
    ap.add_argument("--insert", type=int, default=None, help="K：每两个关键帧之间插K张（scope=local或mode=raft必需）")
    ap.add_argument("--z_axis", type=int, default=2,
                    help="哪一维是层方向：0/1/2（nnUNet常见是2即最后一维）")

    # raft related
    ap.add_argument("--raft_root", type=str, default="", help="RAFT仓库路径（可选，用于sys.path）")
    ap.add_argument("--raft_ckpt", type=str, default="", help="RAFT权重（mode=raft必填）")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--iters", type=int, default=20)

    # naming
    ap.add_argument("--as_modality", action="store_true",
                    help="如果输入名以_0000结尾，则输出自动改为_0001（nnUNet多模态命名）")
    args = ap.parse_args()

    imagesTr = Path(args.imagesTr)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nii_files = sorted([p for p in imagesTr.iterdir() if p.is_file() and (p.suffix in [".nii"] or p.name.endswith(".nii.gz"))],
                       key=lambda x: x.name)
    if len(nii_files) == 0:
        raise RuntimeError(f"{imagesTr} 下没有找到 nii/nii.gz")

    raft_est = None
    if args.mode == "raft":
        if not args.raft_ckpt or args.insert is None:
            raise RuntimeError("mode=raft 必须提供 --raft_ckpt 和 --insert K")
        raft_est = FlowEstimator(args.raft_ckpt, raft_root=args.raft_root, iters=args.iters, device=args.device)

    if args.scope == "local" and args.insert is None:
        raise RuntimeError("scope=local 必须提供 --insert K")

    for f in tqdm(nii_files, desc="Generate Dt"):
        img = nib.load(str(f))
        arr = img.get_fdata(dtype=np.float32)  # 3D
        if arr.ndim != 3:
            raise RuntimeError(f"{f.name} 不是3D，当前 shape={arr.shape}")

        z_axis = args.z_axis
        if z_axis < 0:
            z_axis = arr.ndim + z_axis

        # generate Dt
        if args.mode == "uniform":
            dt = make_dt_uniform(arr.shape, z_axis=z_axis, scope=args.scope, insert_k=args.insert)
        else:
            dt = make_dt_raft(arr, z_axis=z_axis, scope=args.scope, insert_k=args.insert, raft_est=raft_est)

        # output name
        stem = f.name.replace(".nii.gz", "").replace(".nii", "")
        out_stem = infer_out_name(stem, as_modality=args.as_modality)
        out_path = out_dir / f"{out_stem}.nii.gz"

        # save with same affine/header for alignment
        nib.save(nib.Nifti1Image(dt.astype(np.float32), img.affine, img.header), str(out_path))

    print(f"[OK] Done. Dt saved to: {out_dir}")


if __name__ == "__main__":
    main()
