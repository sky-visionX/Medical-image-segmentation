import sys

sys.path.append('../RAFT/')
sys.path.append('../RAFT/core')

import cv2
import torch
import numpy as np
from PIL import Image
from raft import RAFT
from easydict import EasyDict as edict

try:
    from utils import flow_viz
    from utils.utils import InputPadder
except ImportError:
    from RAFT.core.utils import flow_viz
    from RAFT.core.utils.utils import InputPadder


class FlowEstimator:
    """
    This estimator is relied on RAFT
    RAFT: https://github.com/princeton-vl/RAFT
    """

    def __init__(self, checkpoint, iters=20, device='cuda'):
        self.iters = iters
        self.device = device
        args = edict({'mixed_precision': False, 'small': False, 'alternate_corr': False})
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(checkpoint))
        self.model = model.module
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def batch_estimate_flow(self, img1, img2):
        # estimate the optical flow from img1 to img2
        flow_low, flow_up = self.model(img1 * 255., img2 * 255., iters=20, test_mode=True)
        return flow_up

    @torch.no_grad()
    def estimate_flow(self, img1_path, img2_path, visualize=True):
        # estimate the optical flow from img1 to img2
        # return optical flow and visualized rgb of the flow
        img1 = self.load_img(img1_path)
        img2 = self.load_img(img2_path)
        padder = InputPadder(img1.shape)
        img1, img2 = padder.pad(img1, img2)
        flow_low, flow_up = self.model(img1, img2, iters=20, test_mode=True)
        flow_rgb = self.viz(img1, flow_up)
        if visualize:
            cv2.imshow('image', flow_rgb[:, :, [2, 1, 0]] / 255.0)
            cv2.waitKey()
        cv2.imwrite('flow_visualization.png', flow_rgb[:, :, [2, 1, 0]])  # RGB → BGR
        return flow_up, flow_rgb

    def load_img(self, img_path):
        img = np.array(Image.open(img_path)).astype(np.uint8)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)  # 变成 (H, W, 1)
            img = np.repeat(img, 3, axis=2)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(self.device)

    def viz(self, img, flo):
        img = img[0].permute(1, 2, 0).cpu().numpy()
        flo = flo[0].permute(1, 2, 0).cpu().numpy()

        # map flow to rgb image
        flo = flow_viz.flow_to_image(flo)
        img_flo = np.concatenate([img, flo], axis=0)

        return img_flo


def cosine_project_ratio(array1, array2):
    """
    方法1（只加可靠性筛选，不改调用方式）：
    - 对 ||array2|| 很小的地方做 mask，避免除0/爆炸
    - 对 NaN/Inf/越界值做修正
    - 输出稳定在 [0,1]（与你dataset里clip一致）
    """
    # dot(v0t, v01)
    array_inner = array1[..., 0] * array2[..., 0] + array1[..., 1] * array2[..., 1]

    # ||v01||^2
    denom = array2[..., 0] ** 2 + array2[..., 1] ** 2
    mag = np.sqrt(denom + 1e-12)

    # 自适应 eps：用 ||v01|| 的 10% 分位数作为阈值（不需要你定义 eps）
    eps = np.percentile(mag, 10)
    valid = denom > (eps ** 2 + 1e-12)

    # 先按原公式算（只在 valid 区域算，其他先置0）
    array_cos_sim = np.zeros_like(denom, dtype=np.float32)
    array_cos_sim[valid] = (array_inner[valid] / (denom[valid] + 1e-12)).astype(np.float32)

    # clip 到 [0,1]（你dataset里也会clip，这里提前保证稳定）
    array_cos_sim = np.clip(array_cos_sim, 0.0, 1.0)

    # 对 NaN/Inf 再兜底
    array_cos_sim[~np.isfinite(array_cos_sim)] = 0.5

    # 可选：邻域中位数回填（如果你的Dt经常花屏，建议保留；不想用就注释掉）
    if array_cos_sim.ndim == 2:
        med = cv2.medianBlur((array_cos_sim * 255).astype(np.uint8), 5).astype(np.float32) / 255.0
        bad = ~valid
        array_cos_sim[bad] = med[bad]
        array_cos_sim = np.clip(array_cos_sim, 0.0, 1.0)

    return array_cos_sim
    # # calculate the dot product of each pair of vectors in the two arrays
    # array_inner = array1[..., 0] * array2[..., 0] + array1[..., 1] * array2[..., 1]
    # # array1_mag = np.linalg.norm(array1, axis=1)
    # array2_mag = np.linalg.norm(array2, axis=-1)
    # array_cos_sim = array_inner / (array2_mag ** 2)
    #
    # return array_cos_sim


if __name__ == '__main__':
    '''
    cmd:
    python dis_index.py
    '''
    # checkpoint = '../RAFT/models/raft-things.pth'
    # flow_estimator = FlowEstimator(checkpoint=checkpoint)
    #img1_path = '/home/dell/wzh/new/InterpAny-Clearer-main/Meningioma_converted_preprocessedraw/sequences/case_03/im1.png'
    #img2_path = '/home/dell/wzh/new/InterpAny-Clearer-main/Meningioma_converted_preprocessedraw/sequences/case_03/im2.png'
    #img3_path = '/home/dell/wzh/new/InterpAny-Clearer-main/Meningioma_converted_preprocessedraw/sequences/case_03/im3.png'
    # flow_estimator.estimate_flow(img1_path, img2_path, visualize=True)
    # flow_estimator.estimate_flow(img1_path, img3_path, visualize=True)

    array1 = np.array([[1, 2], [3, 4]])
    array2 = np.array([[5, 6], [7, 8]])
    print(cosine_project_ratio(array1, array2))
