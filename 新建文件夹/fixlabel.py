import os
import cv2
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm


def find_reference_file(root_path, case_name):
    """在參考目錄中搜尋原始的 NIfTI 文件以獲取幾何頭信息"""
    for path in Path(root_path).rglob(f"{case_name}.nii.gz"):
        return path
    return None


def fix_label_values(img):
    """將可視化像素 (127, 255) 轉回類別標籤 (1, 2)"""
    new_label = np.zeros_like(img, dtype=np.uint8)
    # 容差處理：防止 PNG 壓縮產生的微小色差
    new_label[np.abs(img.astype(float) - 127) < 10] = 1
    new_label[np.abs(img.astype(float) - 255) < 10] = 2
    return new_label


def warp_label_nearest(lbl1, lbl2, t=0.5):
    """使用光流對標籤進行變形，並保持最近鄰插值以維持類別 ID"""
    # 計算光流 (基於標籤內容)
    flow_12 = cv2.calcOpticalFlowFarneback(lbl1, lbl2, None, 0.5, 3, 15, 3, 5, 1.1, 0)
    flow_21 = cv2.calcOpticalFlowFarneback(lbl2, lbl1, None, 0.5, 3, 15, 3, 5, 1.1, 0)

    h, w = lbl1.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    # 雙向 Warping
    s1x = (grid_x - t * flow_12[..., 0]).astype(np.float32)
    s1y = (grid_y - t * flow_12[..., 1]).astype(np.float32)
    w1 = cv2.remap(lbl1, s1x, s1y, cv2.INTER_NEAREST)

    s2x = (grid_x - (1 - t) * flow_21[..., 0]).astype(np.float32)
    s2y = (grid_y - (1 - t) * flow_21[..., 1]).astype(np.float32)
    w2 = cv2.remap(lbl2, s2x, s2y, cv2.INTER_NEAREST)

    # 融合標籤 (非背景優先)
    res = w1.copy()
    res[w1 == 0] = w2[w1 == 0]
    return res


def process_dataset(processed_root, reference_root, output_dir):
    processed_root = Path(processed_root)
    reference_root = Path(reference_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 獲取所有 case 檔案夾 (如 prostate_00)
    case_folders = sorted([d for d in processed_root.iterdir() if d.is_dir()])

    for case_folder in tqdm(case_folders, desc="正在重建標籤"):
        case_id = case_folder.name

        # 1. 定位原始參考文件
        ref_nii_path = find_reference_file(reference_root, case_id)
        if not ref_nii_path:
            print(f"\n[!] 警告: 找不到 {case_id} 的原始參考文件，將跳過以避免幾何錯誤。")
            continue

        ref_obj = nib.load(str(ref_nii_path))

        all_slices = []
        # 2. 遍歷序列 case_00001, case_00002...
        seq_dir = case_folder / "sequences"
        if not seq_dir.exists(): continue

        case_dirs = sorted(list(seq_dir.glob("case_*")))

        for case in case_dirs:
            # 讀取原始 3 幀
            l1_raw = cv2.imread(str(case / "im1.png"), 0)
            l2_raw = cv2.imread(str(case / "im2.png"), 0)
            l3_raw = cv2.imread(str(case / "im3.png"), 0)

            # --- 步驟 A: 數值校正 ---
            l1 = fix_label_values(l1_raw)
            l2 = fix_label_values(l2_raw)
            l3 = fix_label_values(l3_raw)

            # --- 步驟 B: 插值生成中間層 (3變5邏輯) ---
            m12 = warp_label_nearest(l1, l2)
            m23 = warp_label_nearest(l2, l3)

            # 按順序堆疊
            all_slices += [l1, m12, l2, m23, l3]

        if not all_slices: continue

        # 3. 堆疊成 3D 體 (H, W, D)
        vol = np.stack(all_slices, axis=2).astype(np.uint8)

        # 4. 幾何修復 (幾何對齊的核心)
        header = ref_obj.header.copy()
        affine = ref_obj.affine

        # 修正 Z軸 Spacing：因為層數增加了，必須縮小間距以保持總物理長度不變
        if vol.shape[2] != ref_obj.shape[2]:
            # 公式：新間距 = 舊間距 * (舊層數-1) / (新層數-1)
            factor = (vol.shape[2] - 1) / (ref_obj.shape[2] - 1)
            header['pixdim'][3] = header['pixdim'][3] / factor

        # 5. 保存
        new_nii = nib.Nifti1Image(vol, affine, header)
        nib.save(new_nii, str(output_dir / f"{case_id}.nii.gz"))


if __name__ == "__main__":
    process_dataset(
        processed_root="/home/dell/wzh/new/InterpAny-Clearer-main/datasetlabelprostate",
        reference_root="/home/dell/wzh/data/Task05_Prostate/Task05_Prostate/labelsTr",
        output_dir="/home/dell/wzh/data/Task05_Prostate/prostategai/1"
    )