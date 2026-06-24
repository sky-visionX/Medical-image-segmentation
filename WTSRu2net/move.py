import os
import shutil

# 设置源文件夹和目标文件夹路径
source_folder = '/home/dell/wzh/data/brain/train_masks'  # 替换为源文件夹的路径
target_folder = '/home/dell/wzh/data/brain/val_masks'  # 替换为目标文件夹的路径

# 获取源文件夹中的所有图片文件，按文件名排序
all_images = sorted(os.listdir(source_folder))

# 选择前5876张图片
selected_images = all_images[:10000]#19156

# 确保目标文件夹存在
os.makedirs(target_folder, exist_ok=True)

# 移动图片到目标文件夹
for image in selected_images:
    source_path = os.path.join(source_folder, image)
    target_path = os.path.join(target_folder, image)
    shutil.move(source_path, target_path)

print(f"已成功将 {len(selected_images)} 张图片移动到目标文件夹。")



# import os
# import argparse
# import cv2
# import numpy as np
#
#
# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Filter image-label directories based on matching names and label content."
#     )
#     parser.add_argument(
#         "--label_dir", type=str, help="Path to the label images directory.",
#         default=r"/home/dell/wzh/data/brain/2/volumes"
#     )
#     parser.add_argument(
#         "--img_dir", type=str, help="Path to the input images directory.",
#         default=r"/home/dell/wzh/data/brain/1/volumes"
#     )
#     parser.add_argument(
#         "--delete_unmatched", action='store_true',
#         help="Delete unmatched or content-invalid files when set."
#     )
#     return parser.parse_args()
#
#
# def filter_by_label_content(label_dir, img_dir, delete_unmatched=False):
#     # 列出文件名集合
#     label_files = set(
#         f for f in os.listdir(label_dir)
#         if os.path.isfile(os.path.join(label_dir, f))
#     )
#     img_files = set(
#         f for f in os.listdir(img_dir)
#         if os.path.isfile(os.path.join(img_dir, f))
#     )
#     # 名称匹配
#     matched = label_files & img_files
#
#     valid = set()
#     invalid = set()
#     # 按内容筛选：label图像中是否存在像素 != 1
#     for fname in matched:
#         label_path = os.path.join(label_dir, fname)
#         # 以灰度图方式读取
#         mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
#         if mask is None:
#             print(f"[WARN] 无法读取 {fname}，跳过。")
#             invalid.add(fname)
#             continue
#         # 如果有任意前景像素（值 > 0），则保留
#         if np.any(mask > 0):
#             valid.add(fname)
#         else:
#             invalid.add(fname)
#
#     print(f"名称匹配数量: {len(matched)}")
#     print(f"保留（有前景）: {len(valid)}；剔除（全背景）: {len(invalid)}")
#     # 删除所有无效文件的标签与图像
#     for fname in invalid:
#         for folder in (label_dir, img_dir):
#             path = os.path.join(folder, fname)
#             if os.path.exists(path):
#                 try:
#                     os.remove(path)
#                     print(f"Deleted invalid file: {path}")
#                 except Exception as e:
#                     print(f"Failed deleting {path}: {e}")
#
#     if delete_unmatched:
#         # 删除标签与图像目录中不在 valid 中的文件
#         for fname in (label_files | img_files) - valid:
#             for d in (label_dir, img_dir):
#                 path = os.path.join(d, fname)
#                 if os.path.exists(path):
#                     try:
#                         os.remove(path)
#                     except Exception as e:
#                         print(f"Failed to delete {path}: {e}")
#         print("Deletion complete.")
#     else:
#         print("Delete flag not set, no files were removed.")
#
#     print("Filtering finished. Valid files remain in their original directories.")
#
#
# if __name__ == '__main__':
#     args = parse_args()
#     filter_by_label_content(
#         args.label_dir,
#         args.img_dir,
#         args.delete_unmatched
#     )