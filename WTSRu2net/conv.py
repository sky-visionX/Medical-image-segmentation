import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# 读取图像并转换为灰度
img = cv2.imread('/home/dell/wzh/labels-95-461.jpg', cv2.IMREAD_GRAYSCALE)

# 定义一个简单的卷积核，比如 Sobel 边缘检测核
kernel = np.array([[ -1, -2, -1],
                   [  0,  0,  0],
                   [  1,  2,  1]])

# 进行卷积运算
feature_map = convolve2d(img, kernel, mode='same', boundary='symm')

# 可视化原图和卷积特征图
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(feature_map, cmap='gray')
axes[1].set_title('Convolved Feature Map')
axes[1].axis('off')

# 保存卷积结果
plt.savefig('convolution_feature_map.png', bbox_inches='tight')
plt.show()
