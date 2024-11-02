# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class SoftArgmaxLoss(nn.Module):
    def __init__(self, use_target_weight):
        super(SoftArgmaxLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        H, W = output.size(2), output.size(3)

        grid_x_, grid_y_ = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid_x = grid_x_.reshape(-1).cuda()
        grid_y = grid_y_.reshape(-1).cuda()
        
        heatmaps_pred = output.reshape((batch_size, num_joints, -1))
        heatmaps_gt = target.reshape((batch_size, num_joints, -1))
        loss = 0

        for idx in range(num_joints):
            sam_pred_x = (torch.softmax(20.0 * heatmaps_pred[:,idx,:] - 10.0, dim=-1) * grid_x).sum(-1)
            sam_pred_y = (torch.softmax(20.0 * heatmaps_pred[:,idx,:] - 10.0, dim=-1) * grid_y).sum(-1)
            # sam_gt_x = (torch.softmax(20.0 * heatmaps_gt[:,idx,:] - 10.0, dim=-1) * grid_x).sum(-1)
            # sam_gt_y = (torch.softmax(20.0 * heatmaps_gt[:,idx,:] - 10.0, dim=-1) * grid_y).sum(-1)
            gt_argmax = torch.argmax(heatmaps_gt[:,idx,:], dim=1)
            sam_gt_x = gt_argmax // W
            sam_gt_y = gt_argmax % W
            if self.use_target_weight:
                loss += 0.1 * self.criterion(
                    sam_pred_x.mul(target_weight[:, idx]),
                    sam_gt_x.mul(target_weight[:, idx]))
                loss += 0.1 * self.criterion(
                    sam_pred_y.mul(target_weight[:, idx]),
                    sam_gt_y.mul(target_weight[:, idx]))
            else:
                loss += 0.1 * self.criterion(sam_pred_x, sam_gt_x)
                loss += 0.1 * self.criterion(sam_pred_y, sam_gt_y)

        return loss / num_joints

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)
