from dataset import mydataset
import os
import time
import torch
from torch.optim.lr_scheduler import MultiStepLR
from func.loss import myloss
from torch.utils.data import DataLoader
from model.nnet import Model
# from model.nnet_cord import Model
from tqdm import tqdm
from torch import optim
import numpy as np
from func.plot import plot_loss


def main():
    epochs = 30
    steps = [30, 40]
    batch_size = 15
    lr = 1e-4
    output_dims = 17
    kernel_size = 5
    sigma = 5
    log_root = './log'
    time_temp = time.asctime(time.localtime(time.time())).replace(':', '-')
    log_dir = os.path.join(log_root, time_temp)
    os.makedirs(log_dir, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # COCO大数据集图片
    train_datasets = mydataset(r'E:\mycoco\bigtrain.txt', kernel_size=kernel_size, sigma=sigma)
    val_datasets = mydataset('valid.txt', kernel_size=kernel_size, sigma=sigma)
    # 自制跌倒数据集
    # train_datasets = mydataset('down_train.txt', kernel_size=kernel_size, sigma=sigma)
    # val_datasets = mydataset('down_valid.txt', kernel_size=kernel_size, sigma=sigma)
    train_dataloader = DataLoader(train_datasets, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=3,
                                  persistent_workers=True)
    val_dataloader = DataLoader(val_datasets, batch_size=batch_size, pin_memory=True, num_workers=3,
                                persistent_workers=True)

    Net = Model(output_dims).to(device)
    # 加载模型的参数
    model_path = 'log/140_shuffle_heatmap_down/weights.pt'
    Net.load_state_dict(torch.load(model_path))

    Loss_fun = myloss()
    opt = optim.Adam(Net.parameters(), lr=lr)
    scheduler = MultiStepLR(opt, milestones=steps, gamma=0.1)

    train_loss_list = []
    val_loss_list = []
    train_steps = len(train_dataloader)
    val_steps = len(val_dataloader)

    for epoch in range(epochs):
        Net.train()
        train_per_epoch_loss = torch.zeros(1).cuda()
        train_bar = tqdm(train_dataloader)
        for imgs, heatmaps, viss in train_bar:
            imgs, heatmaps = imgs.to(device), heatmaps.to(device)
            viss = viss.to(device)
            outputs = Net(imgs)
            # loss = Loss_fun(outputs, heatmaps)   #不计算可见性
            loss = Loss_fun(outputs, heatmaps, viss)  # 计算可见性
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_per_epoch_loss += loss.item()

            train_bar.desc = f'epoch[{epoch + 1}/{epochs}]-lr[{opt.param_groups[0]["lr"]:.5f}]-train_loss[{loss.detach():.4f}]'

        train_loss_list.append((train_per_epoch_loss / train_steps).item())

        Net.eval()
        val_per_epoch_loss = torch.zeros(1).cuda()
        with torch.no_grad():
            val_bar = tqdm(val_dataloader)
            for imgs, heatmaps, viss in val_bar:
                imgs, heatmaps = imgs.to(device), heatmaps.to(device)
                viss = viss.to(device)
                outputs = Net(imgs)
                loss = Loss_fun(outputs, heatmaps, viss)
                val_per_epoch_loss += loss
                val_bar.desc = f'epoch[{epoch + 1}/{epochs}]-val_loss[{loss.detach():.4f}]'

            val_loss_list.append((val_per_epoch_loss / val_steps).item())

        scheduler.step()
    plot_loss(log_dir, train_loss_list, val_loss_list)
    np.savetxt(os.path.join(log_dir, 'loss.txt'),
               np.array(train_loss_list + val_loss_list).reshape(-1, 2))
    torch.save(Net.state_dict(), os.path.join(log_dir, 'weights.pt'))


if __name__ == '__main__':
    main()


