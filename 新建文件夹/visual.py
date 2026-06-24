import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_npy_file(npy_file, out_png=None, save_norm_npy=False, cmap='hot'):
    """
    加载 .npy 并可视化，同时支持保存图片（PNG）和可选保存归一化后的npy
    """
    data = np.load(npy_file)
    print(f"Loaded .npy file: {npy_file} with shape: {data.shape}, dtype: {data.dtype}")

    # 选取要显示的二维图
    if data.ndim == 2:
        img = data
    elif data.ndim >= 3:
        # 默认显示第0通道（例如光流u分量）
        img = data[..., 0]
    else:
        raise ValueError(f"Unsupported npy ndim={data.ndim}")

    # 做一个更稳的显示：按分位数拉伸（避免极值导致“全黑/全白”）
    img = img.astype(np.float32)
    vmin = np.percentile(img, 1)
    vmax = np.percentile(img, 99)
    if vmax <= vmin:
        vmax = vmin + 1e-6

    # 归一化到[0,1]（可选保存用）
    img_norm = np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0)

    # 默认输出路径：和npy同目录、同名.png
    if out_png is None:
        base = os.path.splitext(npy_file)[0]
        out_png = base + ".png"

    # 画图
    plt.figure(figsize=(8, 8))
    plt.imshow(img_norm, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar()
    plt.axis('off')
    plt.title(os.path.basename(npy_file))

    # 保存图片（关键：savefig 要放在 show 之前）
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches='tight', pad_inches=0.05)
    print(f"Saved visualization to: {out_png}")

    plt.show()
    plt.close()

    # 可选：把归一化后的二维图也保存成npy（方便后续处理/对比）
    if save_norm_npy:
        out_npy = os.path.splitext(out_png)[0] + "_norm.npy"
        np.save(out_npy, img_norm)
        print(f"Saved normalized array to: {out_npy}")


if __name__ == "__main__":
    npy_file = "/home/dell/wzh/new/InterpAny-Clearer-main/datasetraw/sequences/case_27/dis_index_0_1_2.npy"
    visualize_npy_file(
        npy_file,
        out_png="/home/dell/wzh/new/InterpAny-Clearer-main/datasetraw",
        save_norm_npy=False,   # 想同时保存归一化矩阵就改 True
        cmap="gray"
    )
