import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmseg.registry import MODELS
from mmseg.models.losses.utils import get_class_weight, weighted_loss
import cv2


def sobel(x, kernel, in_channels=1):
    # kernel shape is（out_channels, in_channels, kernel_size, kernel_size）
    # kernel.repeat在原基础上copy倍数
    y = F.conv2d(x, kernel.repeat(1, in_channels, 1, 1), stride=1, padding=1, )
    return y


@weighted_loss
def edge_loss(pred,
              target,
              valid_mask,
              ignore_index,
              **kwards):
    assert pred.shape[0] == target.shape[0]
    batch_size, num_classes, H, W = pred.shape
    seg_pred = pred.argmax(dim=1, keepdim=True)  # (B,C,H,W) to (B,1,H,W)
    seg_target = target.view(batch_size, 1, H, W)  # (B,H,W) to (B,1,H,W)
    seg_pred = (seg_pred != 0).float()  # (B,1,H,W)
    seg_target = (seg_target != 0).float()  # (B,1,H,W)

    # 计算Sobel卷积结果
    # kernel
    sobel_x = torch.tensor([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)
    sobel_x = sobel_x.cuda()
    sobel_y = sobel_y.cuda()

    sobel_x_pred = sobel(seg_pred, sobel_x, 1)
    sobel_y_pred = sobel(seg_pred, sobel_y, 1)
    sobel_x_target = sobel(seg_target, sobel_x, 1)
    sobel_y_target = sobel(seg_target, sobel_y, 1)
    edge_loss = binary_edge_loss(sobel_x_pred, sobel_y_pred, sobel_x_target, sobel_y_target, valid_mask)
    return edge_loss


def binary_edge_loss(x1, y1, x2, y2, valid_mask):
    x1 = x1.reshape(x1.shape[0], -1)
    y1 = y1.reshape(y1.shape[0], -1)
    x2 = x2.reshape(x2.shape[0], -1)
    y2 = y2.reshape(y2.shape[0], -1)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)
    edge_loss = torch.mean((torch.add(torch.abs(x1 - x2), torch.abs(y1 - y2)) * valid_mask), dim=1)
    return edge_loss


@MODELS.register_module()
class EdgeLoss(nn.Module):  # 边缘损失函数
    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 ignore_index=255,
                 loss_name='loss_edge',
                 **kwards):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                avg_factor=None,
                reduction_override=None,
                **kwards) -> Tensor:
        """Forward function.
        Args:
            bd_pre (Tensor): seg_logits [N,C,H,W]. N是batch_size, C是class_num
            bd_gt (Tensor): Ground truth [N,H,W].

        Returns:
            Tensor: Loss tensor.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        valid_mask = (target != self.ignore_index).float()  # (N,H,W)

        loss = self.loss_weight * edge_loss(
            pred,
            target,
            valid_mask=valid_mask,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=self.ignore_index
        )

        return loss

    @property
    def loss_name(self):
        return self._loss_name


# # 测试
# if __name__ == '__main__':
    # pred = cv2.imread('../data/lidianchi3g/train/annotations/lidianchi_0684.png', 0)
    # gt = cv2.imread('../data/lidianchi3g/train/annotations/lidianchi_0164.png', 0)
    # pred = torch.from_numpy(pred)
    # H, W = pred.shape
    # pred = pred.view([1, 1, H, W])
    #
    # gt = torch.from_numpy(gt)
    # H, W = gt.shape
    # gt = gt.view([1, H, W])
    #
    # loss = EdgeLoss()
    # loss = loss.forward(pred, gt)
    # print(loss)
