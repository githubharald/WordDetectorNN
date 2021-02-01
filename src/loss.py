import torch
import torch.nn.functional as F

from coding import MapOrdering


def compute_loss(y, gt_map):
    # 1. segmentation loss
    target_labels = torch.argmax(gt_map[:, MapOrdering.SEG_WORD:MapOrdering.SEG_BACKGROUND + 1], dim=1)
    loss_seg = F.cross_entropy(y[:, MapOrdering.SEG_WORD:MapOrdering.SEG_BACKGROUND + 1], target_labels)

    # 2. geometry loss
    # distances to all sides of aabb
    t = torch.minimum(y[:, MapOrdering.GEO_TOP], gt_map[:, MapOrdering.GEO_TOP])
    b = torch.minimum(y[:, MapOrdering.GEO_BOTTOM], gt_map[:, MapOrdering.GEO_BOTTOM])
    l = torch.minimum(y[:, MapOrdering.GEO_LEFT], gt_map[:, MapOrdering.GEO_LEFT])
    r = torch.minimum(y[:, MapOrdering.GEO_RIGHT], gt_map[:, MapOrdering.GEO_RIGHT])

    # area of predicted aabb
    y_width = y[:, MapOrdering.GEO_LEFT, ...] + y[:, MapOrdering.GEO_RIGHT, ...]
    y_height = y[:, MapOrdering.GEO_TOP, ...] + y[:, MapOrdering.GEO_BOTTOM, ...]
    area1 = y_width * y_height

    # area of gt aabb
    gt_width = gt_map[:, MapOrdering.GEO_LEFT, ...] + gt_map[:, MapOrdering.GEO_RIGHT, ...]
    gt_height = gt_map[:, MapOrdering.GEO_TOP, ...] + gt_map[:, MapOrdering.GEO_BOTTOM, ...]
    area2 = gt_width * gt_height

    # compute intersection over union
    intersection = (r + l) * (b + t)
    union = area1 + area2 - intersection
    eps = 0.01  # avoid division by 0
    iou = intersection / (union + eps)
    iou = iou[gt_map[:, MapOrdering.SEG_WORD] > 0]
    loss_aabb = -torch.log(torch.mean(iou))

    # total loss is simply the sum of both losses
    loss = loss_seg + loss_aabb
    return loss
