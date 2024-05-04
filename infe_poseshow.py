from dataset import *
import numpy as np
from model.nnet import Model
import cv2
from PIL import Image


def main():
    output_dims = 17
    root = 'valid.txt'
    pre_img_list = ['000000000785', '000000000885', r'E:\mycoco\down_data\merged_data_valid\003.jpg', '000000017905']
    # pre_img_list = [r'E:\mycoco\down_data\merged_data_valid\003.jpg', r'E:\mycoco\down_data\merged_data_valid\005.jpg', r'E:\mycoco\down_data\merged_data\072.jpg']
    dataset = open(root, 'r').readlines()
    for i in range(100):
        data_split = dataset[i].split()
        img_path, raw_label = data_split[0], data_split[1:]
        pre_img_list.append(img_path)

    coco_connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8),  # Body
        (8, 10), (5, 11), (6, 12), (11, 12),  # Arms
        (11, 13), (12, 14), (13, 15), (14, 16)  # Legs
    ]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    color_index = 0

    for pre_img in pre_img_list:
        if pre_img == '000000000785':
            img = Image.open(f'E:/mycoco/night_img_valid/{pre_img}.jpg')
        elif pre_img == '000000000885':
            img = Image.open(f'E:/mycoco/night_img_valid/{pre_img}.jpg')
        elif pre_img == '000000017905':
            img = Image.open(f'E:/mycoco/night_img_valid/{pre_img}.jpg')
        else:
            img = Image.open(f'{pre_img}')
        w, h = img.size
        draw_img = np.array(img)
        img = transform(img)
        img = torch.unsqueeze(img, 0)
        Net = Model(output_dims).cuda()
        # Net.load_state_dict(torch.load('log/140_shuffle_heatmap_down/weights.pt'))
        Net.load_state_dict(torch.load('log/Fri May  3 22-44-08 2024/weights.pt'))
        Net.eval()
        outputs = Net(img.cuda()).cpu().detach().numpy()[0]

        labels = []
        scores = []
        for heatmap in outputs:
            max_index = np.argmax(heatmap)
            score = heatmap.flatten()[max_index]
            labels.extend(list([max_index % heatmap_w, max_index // heatmap_h]))
            scores.append(float(score))
        labels = labels + scores
        for i in range(0, 2 * output_dims, 2):
            if labels[2 * output_dims + i // 2] > 0.3:
                draw_img = cv2.circle(draw_img, (int(labels[i] * w / heatmap_w), int(labels[i + 1] * h / heatmap_h)), 5,
                                      (255, 0, 0), -1)

        draw_img = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
        # 添加连接线
        for connection in coco_connections:
            if labels[2 * output_dims + connection[0]] > 0.3 and labels[2 * output_dims + connection[1]] > 0.3:
                pt1 = (
                int(labels[connection[0] * 2] * w / heatmap_w), int(labels[connection[0] * 2 + 1] * h / heatmap_h))
                pt2 = (
                int(labels[connection[1] * 2] * w / heatmap_w), int(labels[connection[1] * 2 + 1] * h / heatmap_h))
                draw_img = cv2.line(draw_img, pt1, pt2, colors[color_index], 2)
                color_index = (color_index + 1) % len(colors)

        cv2.imshow('pred_hm_img', draw_img)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
