import os
import re

# 设置你的文件夹路径
folder_path = "/home/dell/wzh/WTSRu2netsegementation/test_images"  # 替换为你的文件夹路径

# 定义正则表达式匹配模式，捕获数字和后续部分
pattern = r'^volume-(\d+)'
# pattern = r'^volume-(\d+)-(.*)$'

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    old_file_path = os.path.join(folder_path, filename)
    if os.path.isfile(old_file_path):
        # 使用正则表达式匹配文件名
        match = re.match(pattern, filename)
        if match:
            digit = match.group(1)  # 获取数字部分
            # rest = match.group(2)  # 获取后续内容
            new_filename = f"volume-{digit}.jpg"
            # new_filename = f"volume-{digit}-{rest}"
            new_file_path = os.path.join(folder_path, new_filename)

            # 重命名文件
            os.rename(old_file_path, new_file_path)
            print(f"已将 {filename} 重命名为 {new_filename}")

print("Renaming complete.")
