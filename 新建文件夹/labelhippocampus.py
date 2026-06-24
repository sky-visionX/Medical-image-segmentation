import os
import json
import glob
import SimpleITK as sitk


def process_labels_and_json(image_dir, old_label_dir, output_label_dir, json_path):
    """
    image_dir: 已经插完帧的新图像文件夹 (imagesTr)
    old_label_dir: 原始层数的标签文件夹
    output_label_dir: 处理后存放新标签的文件夹
    json_path: 最终生成的 dataset.json 路径
    """
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)

    # 1. 获取所有图像文件
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.nii.gz")))
    training_list = []

    print(f"开始处理，共找到 {len(image_files)} 个文件...")

    for img_path in image_files:
        file_name = os.path.basename(img_path)
        old_label_path = os.path.join(old_label_dir, file_name)
        new_label_path = os.path.join(output_label_dir, file_name)

        if not os.path.exists(old_label_path):
            print(f"跳过: 未找到对应的原始标签 {file_name}")
            continue

        # 读取新图像（作为目标参考）和旧标签
        ref_img = sitk.ReadImage(img_path)
        label = sitk.ReadImage(old_label_path)

        # 2. 执行重采样（插值）
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ref_img)  # 以新图像的尺寸和间距为准
        # 核心：必须使用最近邻插值保持标签值不变
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetOutputDirection(ref_img.GetDirection())
        resampler.SetOutputOrigin(ref_img.GetOrigin())
        resampler.SetOutputSpacing(ref_img.GetSpacing())
        resampler.SetSize(ref_img.GetSize())

        new_label = resampler.Execute(label)

        # 保存新标签
        sitk.WriteImage(new_label, new_label_path)

        # 记录到 JSON 列表中 (使用相对路径)
        training_list.append({
            "image": f"./imagesTr/{file_name}",
            "label": f"./labelsTr/{file_name}"
        })
        print(f"已完成: {file_name}")

    # 3. 构造并写入 dataset.json
    dataset_info = {
        "name": "Hippocampus",
        "description": "Left and right hippocampus segmentation (Resampled)",
        "reference": "Vanderbilt University Medical Center",
        "licence": "CC-BY-SA 4.0",
        "release": "1.1 2026/01/07",
        "tensorImageSize": "3D",
        "modality": {"0": "MRI"},
        "labels": {
            "0": "background",
            "1": "Anterior",
            "2": "Posterior"
        },
        "numTraining": len(training_list),
        "numTest": 0,  # 如果有测试集文件夹，逻辑同上
        "training": training_list
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=4, ensure_ascii=False)

    print(f"\n处理完成！dataset.json 已生成至: {json_path}")


# --- 配置路径 ---
if __name__ == "__main__":
    # 请根据你的实际路径修改：
    IMAGE_TR_DIR = "/home/dell/wzh/data/Task04_Hippocampus/gai/imagesTr"  # 你已经增加完层数的图像路径
    OLD_LABEL_DIR = "/home/dell/wzh/data/Task04_Hippocampus/labelsTr"  # 原始层数的标签路径
    NEW_LABEL_DIR = "/home/dell/wzh/data/Task04_Hippocampus/gai/labelsTr"  # 准备保存新标签的路径
    JSON_SAVE_PATH = "/home/dell/wzh/data/Task04_Hippocampus/gai/dataset.json"  # JSON 保存位置

    process_labels_and_json(IMAGE_TR_DIR, OLD_LABEL_DIR, NEW_LABEL_DIR, JSON_SAVE_PATH)