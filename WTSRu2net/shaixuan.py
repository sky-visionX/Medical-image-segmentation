import os
import numpy as np
from PIL import Image

# 设置文件夹路径
labels_folder = '/home/dell/wzh/WTSRu2netsegementation/train_masks'
images_folder = '/home/dell/wzh/WTSRu2netsegementation/train_images'

# 获取文件夹中所有的标签文件
label_files = [f for f in os.listdir(labels_folder) if f.startswith('labels-')]

# 遍历每个标签文件
for label_file in label_files:
    # 构建对应的图像文件名
    image_file = 'volume-' + label_file[len('labels-'):]

    # 加载标签文件（假设是png格式，若为其他格式可修改）
    label_path = os.path.join(labels_folder, label_file)
    label = np.array(Image.open(label_path))

    # 检查标签是否为空（假设全0表示空标签，具体规则可修改）
    if np.all(label == 0):
        # 如果标签为空，则删除对应的图像文件和标签文件
        image_path = os.path.join(images_folder, image_file)

        # 删除文件
        if os.path.exists(label_path):
            os.remove(label_path)
            print(f"Deleted {label_path}")
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Deleted {image_path}")