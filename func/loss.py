import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDiscretLoss(nn.Module):
    def __init__(self):
        super(KLDiscretLoss, self).__init__()
        self.LogSoftmax = nn.LogSoftmax(dim=1)  # [B,LOGITS]
        self.criterion_ = nn.KLDivLoss(reduction='none')

    def criterion(self, dec_outs, labels):
        scores = self.LogSoftmax(dec_outs)
        # print(scores)
        loss = torch.mean(self.criterion_(scores, labels), dim=1)
        return loss

    def forward(self, output_x, output_y, target_x, target_y, target_weight):
        num_joints = output_x.size(1)
        loss = 0

        for idx in range(num_joints):
            coord_x_pred = output_x[:, idx].squeeze()
            coord_y_pred = output_y[:, idx].squeeze()
            coord_x_gt = target_x[:, idx].squeeze()
            coord_y_gt = target_y[:, idx].squeeze()
            weight = target_weight[:, idx].squeeze()
            loss += (self.criterion(coord_x_pred.float(), coord_x_gt.float()).mul(weight).mean())
            loss += (self.criterion(coord_y_pred.float(), coord_y_gt.float()).mul(weight).mean())
        return loss / num_joints


class myloss(nn.Module):
    def __init__(self, t=0.3):
        super(myloss, self).__init__()
        self.t = t
    def forward(self, outputs, heatmaps, viss=None):
        # # 计算可见性权重
        # [B,N,H,W]
        raw_loss = F.binary_cross_entropy(outputs.to(torch.float32), heatmaps.to(torch.float32), reduce=False)
        # [B,N]
        raw_loss = torch.sum(raw_loss, dim=(2, 3))
        # 最终的损失值
        loss = raw_loss.mul(viss)
        loss = loss.mean()
        # 忽略可见性权重
        # loss = F.binary_cross_entropy(outputs.to(torch.float32), heatmaps.to(torch.float32))

        return loss


class CustomLoss(nn.Module):
    def __init__(self, heatmap_weight=1.0, keypoint_weight=1.0):
        super(CustomLoss, self).__init__()
        self.heatmap_weight = heatmap_weight
        self.keypoint_weight = keypoint_weight

    def forward(self, output, target_heatmap, target_keypoints):
        # 计算热图损失
        heatmap_loss = F.mse_loss(output, target_heatmap)

        # 计算关键点损失
        batchsize, kpts = output.size(0)
        pred_keypoints = torch.argmax(output.view(batchsize, kpts, -1), dim=2)
        keypoint_loss = F.mse_loss(pred_keypoints, target_keypoints)

        # 组合损失
        total_loss = self.heatmap_weight * heatmap_loss + self.keypoint_weight * keypoint_loss

        return total_loss
class mymse(nn.Module):
    def __init__(self, t=0.5):
        super(mymse, self).__init__()
        self.t = t
    def forward(self, outputx, outputy,labelx, labely):

        label_x = labelx.to(torch.float32)
        label_y = labely.to(torch.float32)
        output_x = outputx.to(torch.float32)
        output_y = outputy.to(torch.float32)

        loss = torch.add(F.mse_loss(output_x, label_x), F.mse_loss(output_y, label_y))

        return loss/34