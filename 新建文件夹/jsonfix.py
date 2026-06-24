import os
import json
import glob


def create_nnunet_dataset_json(task_dir, output_json_path):
    """
    task_dir: Task資料夾的根目錄 (例如 Task004_Hippocampus)
    output_json_path: dataset.json 要輸出的路徑
    """
    images_tr_dir = os.path.join(task_dir, 'imagesTr')
    labels_tr_dir = os.path.join(task_dir, 'labelsTr')

    # 1. 獲取訓練集檔案列表
    # nnU-Net v1 通常要求圖像以 .nii.gz 結尾
    img_files = sorted(glob.glob(os.path.join(images_tr_dir, "*.nii.gz")))

    training_pairs = []
    for img_path in img_files:
        file_name = os.path.basename(img_path)

        # 檢查標籤是否存在
        # 注意：如果你的圖像有 _0000.nii.gz 後綴，標籤通常沒有這個後綴，請根據實際情況調整
        label_path = os.path.join(labels_tr_dir, file_name.replace("_0000.nii.gz", ".nii.gz"))

        if os.path.exists(label_path):
            training_pairs.append({
                "image": f"./imagesTr/{file_name}",
                "label": f"./labelsTr/{os.path.basename(label_path)}"
            })
        else:
            print(f"警告: 找不到圖像 {file_name} 對應的標籤，已跳過。")

    # 2. 構造符合規範的字典
    dataset_dict = {
        "name": "Hippocampus",
        "description": "Segmentation of hippocampus after interpolation",
        "reference": "Vanderbilt University Medical Center",
        "licence": "CC-BY-SA 4.0",
        "release": "1.0 2026/01/07",
        "tensorImageSize": "3D",
        "modality": {
            "0": "MRI"
        },
        "labels": {
            "0": "background",
            "1": "Anterior",
            "2": "Posterior"
        },
        "numTraining": len(training_pairs),
        "numTest": 0,  # 沒有測試集則設為 0
        "training": training_pairs,
        "test": []  # 重要：必須包含此空列表，否則會報 KeyError
    }

    # 3. 寫入文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_dict, f, indent=4, ensure_ascii=False)

    print(f"成功生成 dataset.json！")
    print(f"訓練樣本數: {len(training_pairs)}")
    print(f"保存路徑: {output_json_path}")


if __name__ == "__main__":
    # --- 請修改以下路徑 ---
    # TASK_ROOT 是包含 imagesTr 和 labelsTr 的資料夾
    TASK_ROOT = "/home/dell/wzh/data/Task04_Hippocampus/gai"
    SAVE_PATH = os.path.join(TASK_ROOT, "dataset.json")

    create_nnunet_dataset_json(TASK_ROOT, SAVE_PATH)