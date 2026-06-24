import nibabel as nib
import numpy as np
import os
from pathlib import Path

# 修改为你 labelsTr 的实际路径
label_dir = "/home/dell/wzh/data/Task04_Hippocampus/gai/labelsTr"
label_path = Path(label_dir)

print(f"{'文件名':<20} | {'包含的标签值':<20} | {'状态'}")
print("-" * 60)

empty_cases = []

for lbl_file in sorted(list(label_path.glob("*.nii.gz"))):
    img = nib.load(str(lbl_file))
    data = img.get_fdata()
    unique_values = np.unique(data)

    # 转换为列表方便查看
    uv_list = unique_values.tolist()

    # 判断是否只有背景 0
    status = "OK"
    if len(uv_list) == 1 and uv_list[0] == 0:
        status = "!!! EMPTY (只有背景) !!!"
        empty_cases.append(lbl_file.name)
    elif not set([1, 2]).intersection(set(uv_list)):
        # 如果有数值但不是 1 或 2 (比如变成了 127, 255)
        status = "??? 数值错误 (非 0,1,2) ???"
        empty_cases.append(lbl_file.name)

    print(f"{lbl_file.name:<20} | {str(uv_list):<20} | {status}")

print("-" * 60)
print(f"检测完成！共有 {len(empty_cases)} 个文件缺失前景。")


# import nibabel as nib
# import numpy as np
# import os
# from pathlib import Path
# from tqdm import tqdm
#
# # ================= 配置路徑 =================
# # 指向你 nnU-Net raw 資料夾下的 imagesTr
# image_dir = "/home/dell/wzh/new/InterpAny-Clearer-main/1"
#
#
# # ===========================================
#
# def check_images_value(img_dir):
#     img_path = Path(img_dir)
#     if not img_path.exists():
#         print(f"錯誤：找不到路徑 {img_dir}")
#         return
#
#     nii_files = sorted(list(img_path.glob("*.nii.gz")))
#     print(f"找到 {len(nii_files)} 個影像文件。")
#     print("-" * 80)
#     print(f"{'文件名':<25} | {'最大值':<8} | {'最小值':<8} | {'非零像素數':<12} | {'狀態'}")
#     print("-" * 80)
#
#     empty_files = []
#
#     for f in tqdm(nii_files, desc="檢查影像數據"):
#         try:
#             img = nib.load(str(f))
#             data = img.get_fdata()
#
#             max_val = np.max(data)
#             min_val = np.min(data)
#             num_nonzero = np.count_nonzero(data)
#
#             status = "OK"
#             if max_val == 0:
#                 status = "!!! EMPTY (全是0) !!!"
#                 empty_files.append(f.name)
#             elif max_val < 0.0001:  # 數值過小
#                 status = "!!! WARNING (數值接近0) !!!"
#                 empty_files.append(f.name)
#
#             print(f"{f.name:<25} | {max_val:<8.4f} | {min_val:<8.4f} | {num_nonzero:<12} | {status}")
#
#         except Exception as e:
#             print(f"{f.name:<25} | 讀取失敗: {e}")
#
#     print("-" * 80)
#     if empty_files:
#         print(f"檢測完成！共有 {len(empty_files)} 個影像文件存在問題：")
#         for ef in empty_files:
#             print(f" - {ef}")
#     else:
#         print("檢測完成！所有影像似乎都有數據。")
#
#
# if __name__ == "__main__":
#     check_images_value(image_dir)