import torch
import torch.nn as nn


class JointsMSELoss(nn.Module):
    def __init__(self, use_joints_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_joints_weight = use_joints_weight

    def forward(self, output, target, joints_weight):
        batch_size = output.shape[0]
        num_joints = output.shape[1]
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = target[:, idx, ...].reshape(batch_size, -1)
            heatmap_gt = output[:, idx, ...].reshape(batch_size, -1)
            weight = joints_weight[:, idx].reshape(batch_size, -1)

            if self.use_joints_weight:
                loss += self.criterion(
                    torch.mul(heatmap_pred, weight),
                    torch.mul(heatmap_gt, weight)
                )
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints
