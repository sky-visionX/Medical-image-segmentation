import os
import cv2
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 论文处理后的数据根目录 (包含 sequences 文件夹)
PROCESSED_ROOT = "/home/dell/wzh/new/InterpAny-Clearer-main/datasetlabelprostate"
# 2. 原始未处理的 NIfTI 目录 (用于提取原始仿射矩阵)
REFERENCE_ROOT = "/home/dell/wzh/data/Task05_Prostate/Task05_Prostate/labelsTr"
# 3. nnU-Net 目标目录
IMAGES_TR_DIR = "/home/dell/wzh/data/Task05_Prostate/prostategai/imagesTr"
LABELS_TR_DIR = "/home/dell/wzh/data/Task05_Prostate/prostategai/labelsTr"


# ===========================================

def find_reference_file(root_path, case_name):
    """在參考目錄中搜尋原始的 NIfTI 文件以獲取幾何頭信息"""
    for path in Path(root_path).rglob(f"{case_name}.nii.gz"):
        return path
    return None


def fix_label_values(img):
    """將可視化像素 (127, 255) 轉回類別標籤 (1, 2)"""
    new_label = np.zeros_like(img, dtype=np.uint8)
    new_label[np.abs(img.astype(float) - 127) < 10] = 1
    new_label[np.abs(img.astype(float) - 255) < 10] = 2
    return new_label


def warp_label_nearest(lbl1, lbl2, t=0.5):
    """使用光流對標籤進行變形，並保持最近鄰插值"""
    flow_12 = cv2.calcOpticalFlowFarneback(lbl1, lbl2, None, 0.5, 3, 15, 3, 5, 1.1, 0)
    flow_21 = cv2.calcOpticalFlowFarneback(lbl2, lbl1, None, 0.5, 3, 15, 3, 5, 1.1, 0)
    h, w = lbl1.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    s1x = (grid_x - t * flow_12[..., 0]).astype(np.float32)
    s1y = (grid_y - t * flow_12[..., 1]).astype(np.float32)
    w1 = cv2.remap(lbl1, s1x, s1y, cv2.INTER_NEAREST)
    s2x = (grid_x - (1 - t) * flow_21[..., 0]).astype(np.float32)
    s2y = (grid_y - (1 - t) * flow_21[..., 1]).astype(np.float32)
    w2 = cv2.remap(lbl2, s2x, s2y, cv2.INTER_NEAREST)
    res = w1.copy()
    res[w1 == 0] = w2[w1 == 0]
    return res


def process_dataset():
    processed_root = Path(PROCESSED_ROOT)
    reference_root = Path(REFERENCE_ROOT)
    output_dir = Path(LABELS_TR_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_folders = sorted([d for d in processed_root.iterdir() if d.is_dir()])

    # 第一步：重建标签
    for case_folder in tqdm(case_folders, desc="1/2 重建標籤 (3D插值)"):
        case_id = case_folder.name
        ref_nii_path = find_reference_file(reference_root, case_id)
        if not ref_nii_path:
            continue

        ref_obj = nib.load(str(ref_nii_path))
        all_slices = []
        seq_dir = case_folder / "sequences"
        if not seq_dir.exists(): continue

        case_dirs = sorted(list(seq_dir.glob("case_*")))
        for case in case_dirs:
            l1_raw = cv2.imread(str(case / "im1.png"), 0)
            l2_raw = cv2.imread(str(case / "im2.png"), 0)
            l3_raw = cv2.imread(str(case / "im3.png"), 0)

            l1, l2, l3 = fix_label_values(l1_raw), fix_label_values(l2_raw), fix_label_values(l3_raw)
            m12 = warp_label_nearest(l1, l2)
            m23 = warp_label_nearest(l2, l3)
            all_slices += [l1, m12, l2, m23, l3]

        if not all_slices: continue
        vol = np.stack(all_slices, axis=2).astype(np.uint8)

        header = ref_obj.header.copy()
        affine = ref_obj.affine
        if vol.shape[2] != ref_obj.shape[2]:
            factor = (vol.shape[2] - 1) / (ref_obj.shape[2] - 1)
            header['pixdim'][3] = header['pixdim'][3] / factor

        new_nii = nib.Nifti1Image(vol, affine, header)
        nib.save(new_nii, str(output_dir / f"{case_id}.nii.gz"))

    # 第二步：强制图像几何同步
    print("\n开始同步图像几何信息...")
    img_dir = Path(IMAGES_TR_DIR)
    lab_dir = Path(LABELS_TR_DIR)

    # 遍历刚刚生成的标签
    for lab_path in tqdm(list(lab_dir.glob("*.nii.gz")), desc="2/2 同步圖像幾何"):
        case_id = lab_path.name.replace(".nii.gz", "")
        lab_obj = nib.load(str(lab_path))

        # 处理 T2 (_0000) 和 ADC (_0001)
        for suffix in ["_0000", "_0001"]:
            img_path = img_dir / f"{case_id}{suffix}.nii.gz"
            if img_path.exists():
                img_obj = nib.load(str(img_path))

                # 检查层数是否一致
                if img_obj.shape[2] != lab_obj.shape[2]:
                    print(f" [!] 警告: {case_id}{suffix} 层数({img_obj.shape[2]})与标签({lab_obj.shape[2]})不符！")
                    continue

                # 强制图像使用标签的几何信息
                fixed_img = nib.Nifti1Image(img_obj.get_fdata(), lab_obj.affine, lab_obj.header)
                nib.save(fixed_img, str(img_path))

    print("\n[完成] 标签重建与图像几何同步已全部结束。")


if __name__ == "__main__":
    process_dataset()