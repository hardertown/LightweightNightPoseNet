from dataset import *
import numpy as np
from model.nnet import Model
# from model.nnet_cord import Model
import cv2
from PIL import Image


def calculate_oks(predicted_keypoints, true_keypoints, sigma=0.05):
    num_keypoints = len(predicted_keypoints)
    oks_sum = 0
    sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
    num_visible_keypoints = 0
    ap = 0
    for i, (pred_keypoint, true_keypoint) in enumerate(zip(predicted_keypoints, true_keypoints), 0):
        distance = np.linalg.norm(np.array(pred_keypoint[:2]) - np.array(true_keypoint[:2]))
        oks_sum += np.exp(-distance ** 2 / (2 * w * h * sigmas[i] ** 2)) * (true_keypoint[2] >= 0.9)  # COCO参数
        # oks_sum += np.exp(-distance ** 2 / (2 * w * h * sigma ** 2)) * (true_keypoint[2] >= 0.9)  # 平均参数
        num_visible_keypoints += true_keypoint[2] >= 0.9
        if np.exp(-distance ** 2 / (2 * w * h * sigmas[i] ** 2)) > 0.5:
            ap = ap + 1
    oks = oks_sum / max(1, num_visible_keypoints)
    return oks, ap/17


def get_(img):
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    Net = Model(17).cuda()
    # Net.load_state_dict(torch.load('log/140_shuffle_heatmap_down/weights.pt'))
    Net.load_state_dict(torch.load('log/Fri May  3 22-44-08 2024/weights.pt'))
    Net.eval()
    outputs = Net(img.cuda()).cpu().detach().numpy()[0]
    labels = []
    scores = []
    for heatmap in outputs:
        max_index = np.argmax(heatmap)
        # print(max_index)
        score = heatmap.flatten()[max_index]
        labels.extend(list([max_index % heatmap_w, max_index // heatmap_h]))
        scores.append(float(score))
    # print(labels)
    labels = labels + scores
    pred_points = []
    for i in range(0, 34, 2):
        pred_points.append((int(labels[i]*w/heatmap_w), int(labels[i+1]*h/heatmap_h), labels[34+i//2]))
    return pred_points

# 获取真实关键点
def get_true_keypoints(raw_label):
    keypoints = [(int(raw_label[i]), int(raw_label[i+1]), int(raw_label[i+2])//2) for i in range(0, len(raw_label), 3)]
    return keypoints

root = 'valid.txt'
# root = 'down_valid.txt'
dataset = open(root, 'r').readlines()
ap_mean = 0
oks_mean = 0
for i in range(344):
    data_split = dataset[i].split()
    img_path, raw_label = data_split[0], data_split[1:]
    imgs = Image.open(f'{img_path}')
    w, h = imgs.size
    true_keypoints = get_true_keypoints(raw_label)
    predicted_keypoints = get_(imgs)
    # print(true_keypoints)
    # print(predicted_keypoints)
    oks_score, ap = calculate_oks(predicted_keypoints, true_keypoints)
    # print("OKS:", oks_score)
    oks_mean += oks_score
    ap_mean += ap
print("Mean OKS:", oks_mean / 344)
print("Mean AP:", ap_mean / 344)
