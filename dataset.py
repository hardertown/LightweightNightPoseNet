from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from func.loss import KLDiscretLoss
from model.lnnet import Model
import torch
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
target_w = 224
target_h = 224
heatmap_w = 56
heatmap_h = 56

transform = transforms.Compose([transforms.Resize((target_h, target_w)),
                                transforms.ToTensor(),
                                transforms.Normalize(0.5, 0.5),
                                ])


class mydataset(Dataset):
    def __init__(self, root, kernel_size=5, sigma=5):
        self.dataset = open(root, 'r').readlines()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.stride = kernel_size // 2
        self.gauss_kernel = self.gauss(self.kernel_size, self.sigma)

    def __len__(self):
        return len(self.dataset)

    def gauss(self, kernel_size, sigma):
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        s = 2 * sigma ** 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                x, y = i - center, j - center
                kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / s)
        kernel = kernel / kernel.max()
        return kernel

    def __getitem__(self, index):
        data_split = self.dataset[index].split()
        img_path, raw_label = data_split[0], data_split[1:]
        img = Image.open(img_path)
        w, h = img.size
        img = transform(img)
        vis = []
        heatmap = np.zeros((len(raw_label)//3, heatmap_h+2*self.stride, heatmap_w+2*self.stride))
        # print(heatmap.shape)
        for i in range(0, len(raw_label), 3):
            if int(raw_label[i+2]) != 0:
                cx = int(float(raw_label[i])*heatmap_w/w) + self.stride
                cy = int(float(raw_label[i+1])*heatmap_h/h) + self.stride
                heatmap[i//3, cy-self.stride:cy+self.stride+1, cx-self.stride:cx+self.stride+1] = self.gauss_kernel
            vis.append(int(raw_label[i+2])/2)

        # print(vis)
        # print(self.stride, target_h-self.stride)
        heatmap = heatmap[:, self.stride: heatmap_h+self.stride, self.stride: heatmap_w+self.stride]
        return img, torch.as_tensor(heatmap), torch.as_tensor(vis)


class mydataset_simcc(Dataset):
    def __init__(self, root):
        self.dataset = open(root, 'r').readlines()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_split = self.dataset[index].split()
        img_path, raw_label = data_split[0], data_split[1:]
        img = Image.open(img_path)
        w, h = img.size
        img = transform(img)
        vis = []
        target_x = np.zeros((17, 224))
        target_y = np.zeros((17, 224))
        target_weight = np.zeros((17, 1))
        j = 0
        for i in range(0, len(raw_label), 3):
            if int(raw_label[i+2]) != 0:
                cx = int(float(raw_label[i])*224/w)
                cy = int(float(raw_label[i+1])*224/h)
                x = np.arange(0, int(224), 1, np.float32)
                y = np.arange(0, int(224), 1, np.float32)
                target_x[j] = (np.exp(- ((x - cx) ** 2) / (2 * 1 ** 2))) / (1 * np.sqrt(np.pi * 2))
                target_y[j] = (np.exp(- ((y - cy) ** 2) / (2 * 1 ** 2))) / (1 * np.sqrt(np.pi * 2))
                j = j+1
            vis.append(int(raw_label[i + 2])/2)
        target_weight[:, 0] = vis[:]
        target_x = torch.from_numpy(target_x)
        target_y = torch.from_numpy(target_y)
        target_weight = torch.from_numpy(target_weight)
        return img, target_x, target_y, target_weight


class mydataset_simcc_reg(Dataset):
    def __init__(self, root):
        self.dataset = open(root, 'r').readlines()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_split = self.dataset[index].split()
        img_path, raw_label = data_split[0], data_split[1:]
        img = Image.open(img_path)
        w, h = img.size
        img = transform(img)
        vis = []
        target_x = np.zeros(17)
        target_y = np.zeros(17)
        target_weight = np.zeros((17, 1))
        j = 0
        for i in range(0, len(raw_label), 3):
            if int(raw_label[i+2]) != 0:
                cx = int(float(raw_label[i])*224/w)
                cy = int(float(raw_label[i+1])*224/h)
                target_x[j] = cx
                target_y[j] = cy
                j = j+1
            vis.append(int(raw_label[i + 2])/2)
        target_weight[:, 0] = vis[:]
        return img, target_x, target_y, target_weight


if __name__ == '__main__':
    train_datasets = mydataset_simcc('./train1.txt')
    train_dataloader = DataLoader(train_datasets, batch_size=2)
    model = Model()
    model.load_state_dict(torch.load('log_simcc/210_noES_nodown/weights.pt'))
    Loss_fun = KLDiscretLoss()
    for (img, target_x, target_y, target_weight) in train_dataloader:

        # 观察x的高斯图
        # print(target_x.shape)
        # plt.matshow(target_x[0])
        # plt.show()
        # # 误差求取
        pred_x, pred_y = model(img)
        # print(target_x)
        # plt.matshow(pred_x[0].cpu().detach().numpy())
        # plt.show()
        loss = Loss_fun(pred_x, pred_y, target_x, target_y, target_weight)
        print(loss.item())

