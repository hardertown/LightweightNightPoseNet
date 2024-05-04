from dataset import *
import numpy as np
from model.lnnet import Model
# from model.lnnet_mobile_se import Model
import cv2
from PIL import Image
import torch.nn as nn


def main():

    # pre_img_list = [r'image_0.jpg']
    pre_img_list = ['000000000785', '000000000885', 'image_0.jpg', '000000378482','000000173799']
    # pre_img_list = [r'E:\mycoco\down_data\merged_data_valid\003.jpg', r'E:\mycoco\down_data\merged_data_valid\005.jpg', r'E:\mycoco\down_data\merged_data\072.jpg']
    for pre_img in pre_img_list:
        if pre_img == 'image_0.jpg':
            img = Image.open(f'{pre_img}')
        elif pre_img =='000000378482':
            img = Image.open(f'E:/mycoco/night_img_train/{pre_img}.jpg')
        # else:
        #     img = Image.open(f'{pre_img}')
        else:
            img = Image.open(f'E:/mycoco/night_img_valid/{pre_img}.jpg')
        w, h = img.size
        draw_img = np.array(img)
        img = transform(img)
        img = torch.unsqueeze(img, 0)
        Net = Model().cuda()
        Net.load_state_dict(torch.load('log_simcc/210_noES_nodown/weights.pt'))
        # Net.load_state_dict(torch.load('log_simcc/210_ES_nodown_mobile/weights.pt'))
        # Net.load_state_dict(torch.load('log_simcc/140_shuffle_down/weights.pt'))
        Net.eval()
        output = Net(img.cuda())
        # 将每个张量移回 CPU 并转换为 NumPy 数组
        pred_x = output[0].cpu().detach().numpy()
        pred_y = output[1].cpu().detach().numpy()
        pred_x = pred_x[0]
        pred_y = pred_y[0]

        # KL散度误差回归
        # 显示高斯图
        # plt.matshow(pred_x[0])
        # plt.show()
        # 沿着第二个维度（沿着宽度）找到最大值的索引, simcc_predict
        max_indices_x = np.argmax(pred_x, axis=1)
        max_indices_y = np.argmax(pred_y, axis=1)
        # 将索引转换为坐标
        coords = [(int(max_indices_x[i]*w/224), int(max_indices_y[i]*h/224)) for i in range(17)]
        print(coords)

        #  sincc_reg, softmax回归
        # coord = np.zeros((17, 2))
        # for i in range(224):
        #     for j in range(17):
        #         coord[j, 0] += i*pred_x[j, i]
        #         coord[j, 1] += i*pred_y[j, i]
        # print(coord.shape)
        # coords = [(int(coord[i, 0]*w/224), int(coord[i, 1]*h/224))for i in range(17)]
        # print(coords)
        for i in range(17):
            draw_img = cv2.circle(draw_img, coords[i], 5, (255, 0, 0), -1)
        draw_img = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
        cv2.imshow('pred_hm_img', draw_img)
        cv2.waitKey(0)



if __name__ == '__main__':
    main()