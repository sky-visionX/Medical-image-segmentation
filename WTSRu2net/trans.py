from PIL import Image
import os

# 设置源文件夹路径
source_folder = '/home/dell/wzh/WTSRu2netsegementation/train_images'  # 替换为源文件夹的路径

# 获取文件夹中的所有.png文件
png_files = [f for f in os.listdir(source_folder) if f.endswith('.png')]

# 遍历每个.png文件并转换为.jpg格式/home/dell/wzh/WTSRu2netsegementation/train_images
for png_file in png_files:
    png_path = os.path.join(source_folder, png_file)
    jpg_path = os.path.join(source_folder, png_file.replace('.jpeg', '.jpg'))

    # 打开PNG图片并保存为JPG格式
    with Image.open(png_path) as img:
        rgb_img = img.convert('RGB')  # 转换为RGB模式以便保存为JPG
        rgb_img.save(jpg_path, 'JPEG')

    # 可选：删除原始PNG文件
    # os.remove(png_path)
    os.remove(png_path)
print(f"已成功将 {len(png_files)} 张PNG图片转换为JPG格式。")
