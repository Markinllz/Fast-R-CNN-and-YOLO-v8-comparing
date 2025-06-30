from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors, dist2bbox, bbox2dist
from ultralytics.utils.loss import BboxLoss
import torch
import torch.nn as nn
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import nms


def generate_anchors(preds, strides):
    anchor_points, stride_tensor = make_anchors(preds , strides)
    return anchor_points, stride_tensor


def process_outputs(preds, num_classes = 2, reg_max = 16):
    cls_preds, dfl_preds = [], []

    for p in preds:
        B, C, H, W = p.shape
        N = H*W

        cls_preds.append(p[:, :num_classes, :, :].permute(0,2,3,1).reshape(B,N,num_classes))
        dfl_preds.append(p[:,num_classes:, :, :].permute(0,2,3,1).reshape(B,N,4*reg_max))


    cls_preds = torch.cat(cls_preds, dim = 1)
    dlf_preds = torch.cat(dfl_preds, dim = 1)

    return cls_preds, dlf_preds


def prepare_targets(targets, B, device, num_classes = 2):
    max_num_gt = max([t.shape[0] for t in targets]) if targets else 1
    gt_labels = torch.full((B, max_num_gt, 1), -1,device=device, dtype=torch.long)
    gt_bboxes = torch.zeros((B, max_num_gt, 4), device=device)
    mask_gt = torch.zeros((B, max_num_gt, 1), device=device, dtype=torch.bool)

    for i , t in enumerate(targets):
        n = t.shape[0]

        if n > 0:

            gt_labels[i, :n, 0] = t[:, 0].long().to(device)
            xc, yc, w, h = t[:, 1], t[:, 2], t[:, 3], t[:, 4]
            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2
            gt_bboxes[i, :n, 0] = x1
            gt_bboxes[i, :n, 1] = y1
            gt_bboxes[i, :n, 2] = x2
            gt_bboxes[i, :n, 3] = y2
            mask_gt[i, :n, 0] = 1

    return  gt_labels, gt_bboxes, mask_gt



def decode_dlf(dfl_preds, anchor_points, reg_max = 16):
    B, N, _ = dfl_preds.shape
    dfl_preds_ = dfl_preds.view(B, N, 4 , reg_max)
    dfl_probs = torch.softmax(dfl_preds_, dim=-1)
    proj = torch.arange(reg_max, device=dfl_preds_.device, dtype=dfl_preds_.dtype)
    ltrb = (dfl_probs*proj).sum(-1)
    bboxes = bbox2dist(anchor_points,ltrb,reg_max)

    return bboxes



def assign_predictions_to_targets(cls_preds, dfl_preds, anchor_points, gt_labels, gt_bboxes, mask_gt, num_classes=2, reg_max=16, device='cpu'):

    assigner = TaskAlignedAssigner(topk=10, num_classes=num_classes)
    dfl_preds_reshaped = dfl_preds.view(dfl_preds.shape[0], dfl_preds.shape[1], 4, reg_max)


    assigned_labels, assigned_bboxes, assigned_scores, fg_mask, target_gt_idx = assigner(
        cls_preds, dfl_preds_reshaped, anchor_points, gt_labels, gt_bboxes, mask_gt
    )


    return assigned_labels, assigned_bboxes, assigned_scores, fg_mask, target_gt_idx




class YoloLoss:
    def __init__(self, num_classes=2, reg_max=16, device='cpu', lambda_cls = 0.5, lambda_dfl =1.5 , lambda_bbox = 7.5):
        self.bbox_loss = BboxLoss()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.device = device
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.lambda_cls = lambda_cls
        self.lambda_bbox = lambda_bbox
        self.lambda_dfl = lambda_dfl

        

    def __call__(self, cls_preds, dfl_preds, assigned_labels, assigned_bboxes, assigned_scores, fg_mask):
        pos_mask = fg_mask.bool()
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device), {'loss_cls': 0.0, 'loss_bbox': 0.0, 'loss_dfl': 0.0}
        


        cls_preds_pos = cls_preds[fg_mask]
        assigned_labels_pos = assigned_labels[fg_mask].long().squeeze(-1)
        cls_loss = self.ce_loss(cls_preds_pos, assigned_labels_pos).mean()
        dfl_preds_pos = dfl_preds[pos_mask]
        assigned_bboxes_pos = assigned_bboxes[pos_mask]
        assigned_scores_pos = assigned_scores[pos_mask]

        loss_bbox, loss_dfl = self.bbox_loss(
            pred_bboxes=None,
            target_bboxes=assigned_bboxes_pos,
            pred_dfl=dfl_preds_pos,
            target_dfl=assigned_bboxes_pos,
            mask=None
        )


        total_loss = self.lambda_cls*cls_loss + self.lambda_bbox * loss_bbox + self.lambda_dfl*loss_dfl


        return total_loss, {
            'loss_cls': cls_loss.item(),
            'loss_bbox': loss_bbox.item(),
            'loss_dfl': loss_dfl.item()
        }





def convert_to_preds(cls_preds, dfl_preds, anchor_points, reg_max = 16, conf_thres=0.25, iou_thres=0.5, max_det=300):

    B, N, num_classes = cls_preds.shape
    results = []


    for b in range(B):
        probs = torch.softmax(cls_preds[b], dim =-1)
        scores, labels = probs.max(dim =-1)

        boxes = decode_dlf(dfl_preds[b].unsqueeze(0),anchor_points=anchor_points, reg_max=reg_max)[0]


        mask = scores > conf_thres
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]


        if boxes.shape[0] > 0:
            keep = nms(boxes, scores, iou_thres)
            keep = keep[:max_det]
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
        else:
            boxes = torch.zeros((0, 4), device=boxes.device)
            scores = torch.zeros((0,), device=boxes.device)
            labels = torch.zeros((0,), dtype=torch.long, device=boxes.device)


        results.append({
            'boxes': boxes,
            'scores': scores,
            'labels': labels
        })
    return results


def convert_targets_for_map(targets):
    batch_targets = []
    for t in targets:
        if t.shape[0] == 0:
            batch_targets.append({'boxes': torch.zeros((0, 4), device=t.device), 'labels': torch.zeros((0,), dtype=torch.long, device=t.device)})
            continue
        xc, yc, w, h = t[:, 1], t[:, 2], t[:, 3], t[:, 4]
        x1 = xc - w / 2
        y1 = yc - h / 2
        x2 = xc + w / 2
        y2 = yc + h / 2
        boxes = torch.stack([x1, y1, x2, y2], dim=1)
        labels = t[:, 0].long()
        batch_targets.append({'boxes': boxes, 'labels': labels})
    return batch_targets


def compute_map(results, targets):
    mAP_metric = MeanAveragePrecision()
    mAP_metric.update(results, targets)
    return mAP_metric.compute()








        