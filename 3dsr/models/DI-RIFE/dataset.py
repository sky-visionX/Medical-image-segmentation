import os
import cv2
import ast
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset

cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VimeoDataset(Dataset):
    def __init__(self, dataset_name, batch_size=32):
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.h = 256
        self.w = 256
        self.data_root = '/home/dell/wzh/new/InterpAny-Clearer-main/Meningioma_converted_preprocessedraw/sequences'
        self.image_root = os.path.join(self.data_root, 'sequences')
        train_fn = os.path.join(self.data_root, 'tri_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'tri_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()
        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        cnt = int(len(self.trainlist) * 0.95)
        if self.dataset_name == 'train':
            self.meta_data = self.trainlist[:cnt]
        elif self.dataset_name == 'test':
            self.meta_data = self.testlist
        else:
            self.meta_data = self.trainlist[cnt:]

    def crop(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x + h, y:y + w, :]
        img1 = img1[x:x + h, y:y + w, :]
        gt = gt[x:x + h, y:y + w, :]
        return img0, gt, img1

    def getimg(self, index):
        imgpath = os.path.join(self.image_root, self.meta_data[index])
        imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png']

        # Load images
        img0 = cv2.imread(imgpaths[0])
        gt = cv2.imread(imgpaths[1])
        img1 = cv2.imread(imgpaths[2])
        timestep = 0.5
        return img0, gt, img1, timestep

        # RIFEm with Vimeo-Septuplet
        # imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png', imgpath + '/im4.png', imgpath + '/im5.png', imgpath + '/im6.png', imgpath + '/im7.png']
        # ind = [0, 1, 2, 3, 4, 5, 6]
        # random.shuffle(ind)
        # ind = ind[:3]
        # ind.sort()
        # img0 = cv2.imread(imgpaths[ind[0]])
        # gt = cv2.imread(imgpaths[ind[1]])
        # img1 = cv2.imread(imgpaths[ind[2]])        
        # timestep = (ind[1] - ind[0]) * 1.0 / (ind[2] - ind[0] + 1e-6)

    def __getitem__(self, index):
        img0, gt, img1, timestep = self.getimg(index)
        if self.dataset_name == 'train':
            img0, gt, img1 = self.crop(img0, gt, img1, 224, 224)
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]
            if random.uniform(0, 1) < 0.5:
                tmp = img1
                img1 = img0
                img0 = tmp
                timestep = 1 - timestep
            # random rotation
            p = random.uniform(0, 1)
            if p < 0.25:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
            elif p < 0.5:
                img0 = cv2.rotate(img0, cv2.ROTATE_180)
                gt = cv2.rotate(gt, cv2.ROTATE_180)
                img1 = cv2.rotate(img1, cv2.ROTATE_180)
            elif p < 0.75:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        timestep = torch.tensor(timestep).reshape(1, 1, 1)
        return torch.cat((img0, img1, gt), 0), timestep


class MRIDataset(Dataset):
    def __init__(self, data_root, transform=None, axis=2, crop_size=(224, 224)):
        self.data_root = data_root
        self.transform = transform
        self.axis = axis
        self.crop_size = crop_size
        self.file_list = [os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith('.nii.gz')]

    def __len__(self):
        return len(self.file_list)

    def _load_and_slice(self, file_path):
        img = nib.load(file_path)
        data = img.get_fdata()
        slices = [data.take(i, axis=self.axis) for i in range(data.shape[self.axis])]
        return slices

    def _random_crop(self, img0, gt, img1):
        h, w = self.crop_size
        ih, iw = img0.shape[:2]
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w]
        img1 = img1[x:x+h, y:y+w]
        gt = gt[x:x+h, y:y+w]
        return img0, gt, img1

    def _augment(self, img0, img1, gt):
        # 随机翻转
        if random.uniform(0, 1) < 0.5:
            img0 = img0[::-1]
            img1 = img1[::-1]
            gt = gt[::-1]
        if random.uniform(0, 1) < 0.5:
            img0 = img0[:, ::-1]
            img1 = img1[:, ::-1]
            gt = gt[:, ::-1]
        # 颜色反转
        if random.uniform(0, 1) < 0.5:
            img0 = 255. - img0
            img1 = 255. - img1
            gt = 255. - gt
        return img0, img1, gt

    def __getitem__(self, index):
        file_path = self.file_list[index]
        slices = self._load_and_slice(file_path)

        # 随机选择连续三个切片
        idx = random.randint(1, len(slices) - 2)
        img0 = slices[idx - 1]
        gt = slices[idx]
        img1 = slices[idx + 1]

        # 归一化
        img0 = (img0 - img0.min()) / (img0.max() - img0.min())
        img1 = (img1 - img1.min()) / (img1.max() - img1.min())
        gt = (gt - gt.min()) / (gt.max() - gt.min())

        # 转换为 uint8
        img0 = (img0 * 255).astype(np.uint8)
        img1 = (img1 * 255).astype(np.uint8)
        gt = (gt * 255).astype(np.uint8)

        # 随机裁剪和增强
        img0, gt, img1 = self._random_crop(img0, gt, img1)
        img0, img1, gt = self._augment(img0, img1, gt)

        # 转为 Tensor
        img0 = torch.from_numpy(img0.copy()).unsqueeze(0).float()  # (1, H, W)
        img1 = torch.from_numpy(img1.copy()).unsqueeze(0).float()
        gt = torch.from_numpy(gt.copy()).unsqueeze(0).float()

        # 时间戳
        timestep = torch.tensor(0.5).reshape(1, 1, 1)

        return torch.cat((img0, img1, gt), 0), timestep