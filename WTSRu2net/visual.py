import cv2
import pywt
import matplotlib.pyplot as plt

# 读取图像并转换为灰度
img = cv2.imread('/home/dell/wzh/WTSRu2netsegementation/val_masks/labels-95-461.jpg', cv2.IMREAD_GRAYSCALE)

# 进行单层小波变换
coeffs2 = pywt.dwt2(img, 'haar')
LL, (LH, HL, HH) = coeffs2

# 2x2 可视化
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
fig.suptitle('Wavelet Sub-bands (2x2 Layout)')

axes[0, 0].imshow(LL, cmap='gray')
axes[0, 0].set_title('LL')
axes[0, 0].axis('off')

axes[0, 1].imshow(LH, cmap='gray')
axes[0, 1].set_title('LH')
axes[0, 1].axis('off')

axes[1, 0].imshow(HL, cmap='gray')
axes[1, 0].set_title('HL')
axes[1, 0].axis('off')

axes[1, 1].imshow(HH, cmap='gray')
axes[1, 1].set_title('HH')
axes[1, 1].axis('off')

# 保存图像
plt.savefig('wavelet_subbands.png', bbox_inches='tight')
plt.show()
