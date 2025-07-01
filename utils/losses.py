from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors, dist2bbox, bbox2dist
from ultralytics.utils.loss import BboxLoss
import torch
import torch.nn as nn
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import nms


def generate_anchors(preds, strides):
  
    anchor_points = []
    stride_tensor = []
    
    for pred, stride in zip(preds, strides):
        B, C, H, W = pred.shape
        
       
        yv, xv = torch.meshgrid([torch.arange(H), torch.arange(W)], indexing='ij')
        grid = torch.stack((xv, yv), 2).float()
        
     
        grid = (grid + 0.5) * stride
        
       
        anchor_points.append(grid.reshape(-1, 2))
        stride_tensor.append(torch.full((H*W,), stride, dtype=torch.float32))
    
    return torch.cat(anchor_points, 0), torch.cat(stride_tensor, 0)


def process_outputs(preds, num_classes=2, reg_max=16):
    cls_preds, dfl_preds = [], []

    for p in preds:
        B, C, H, W = p.shape
        N = H*W

        cls_preds.append(p[:, :num_classes, :, :].permute(0,2,3,1).reshape(B,N,num_classes))
        dfl_preds.append(p[:,num_classes:, :, :].permute(0,2,3,1).reshape(B,N,4*reg_max))


    cls_preds = torch.cat(cls_preds, dim=1)
    dfl_preds = torch.cat(dfl_preds, dim=1)

    return cls_preds, dfl_preds


def prepare_targets(targets, B, device, num_classes=2):
    max_targets = max(len(t) for t in targets) if targets else 1
    
    gt_labels = torch.zeros(B, max_targets, 1, device=device)
    gt_bboxes = torch.zeros(B, max_targets, 4, device=device)
    mask_gt = torch.zeros(B, max_targets, 1, device=device, dtype=torch.bool)
    
    for i, t in enumerate(targets):
        if len(t) == 0:
            continue
        n = len(t)
        if n > max_targets:
            t = t[:max_targets]
            n = max_targets
            
        gt_labels[i, :n, 0] = t[:, 0]
        
      
        if t.shape[1] >= 5:
            xc, yc, w, h = t[:, 1], t[:, 2], t[:, 3], t[:, 4]
            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2
            gt_bboxes[i, :n, 0] = x1
            gt_bboxes[i, :n, 1] = y1
            gt_bboxes[i, :n, 2] = x2
            gt_bboxes[i, :n, 3] = y2
            mask_gt[i, :n, 0] = True

    return gt_labels, gt_bboxes, mask_gt


def decode_dfl(dfl_preds, anchor_points, reg_max=16):
   
    B, N, _ = dfl_preds.shape
    dfl_preds_ = dfl_preds.view(B, N, 4, reg_max)
    dfl_probs = torch.softmax(dfl_preds_, dim=-1)
    proj = torch.arange(reg_max, device=dfl_preds_.device, dtype=dfl_preds_.dtype)
    ltrb = (dfl_probs * proj).sum(-1)  # [B, N, 4]
    
   
    anchor_points = anchor_points.to(dfl_preds.device)
    
   
    if anchor_points.dim() == 2:
        anchor_points = anchor_points.unsqueeze(0).expand(B, -1, -1)  # [B, N, 2]
    
    
    x1 = anchor_points[..., 0:1] - ltrb[..., 0:1]  # left
    y1 = anchor_points[..., 1:2] - ltrb[..., 1:2]  # top  
    x2 = anchor_points[..., 0:1] + ltrb[..., 2:3]  # right
    y2 = anchor_points[..., 1:2] + ltrb[..., 3:4]  # bottom
    
    bboxes = torch.cat([x1, y1, x2, y2], dim=-1)  # [B, N, 4]
    
    return bboxes


def assign_predictions_to_targets(cls_preds, dfl_preds, anchor_points, gt_labels, gt_bboxes, mask_gt, num_classes=2, reg_max=16, device='cpu'):
    
    print(f"cls_preds shape: {cls_preds.shape}")
    print(f"dfl_preds shape: {dfl_preds.shape}")
    print(f"anchor_points shape: {anchor_points.shape}")
    print(f"gt_labels shape: {gt_labels.shape}")
    print(f"gt_bboxes shape: {gt_bboxes.shape}")
    print(f"mask_gt shape: {mask_gt.shape}")
   
    assigner = TaskAlignedAssigner(topk=10, num_classes=num_classes)
    

    with torch.no_grad():
        pd_scores = cls_preds.sigmoid()
        print(f"pd_scores shape: {pd_scores.shape}")
     
        pd_bboxes = decode_dfl(dfl_preds, anchor_points, reg_max)
        print(f"pd_bboxes shape: {pd_bboxes.shape}")
        
        print(f"gt_labels original shape: {gt_labels.shape}")
        print(f"mask_gt original shape: {mask_gt.shape}")

      
        assigned_labels, assigned_bboxes, assigned_scores, fg_mask, target_gt_idx = assigner(
            pd_scores, pd_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt
        )

    return assigned_labels, assigned_bboxes, assigned_scores, fg_mask, target_gt_idx


class YoloLoss:
    def __init__(self, num_classes=2, reg_max=16, device='cpu', lambda_cls=0.5, lambda_dfl=1.5, lambda_bbox=7.5):
        self.bbox_loss = BboxLoss()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.device = device
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.lambda_cls = lambda_cls
        self.lambda_bbox = lambda_bbox
        self.lambda_dfl = lambda_dfl

    def __call__(self, cls_preds, dfl_preds, assigned_labels, assigned_bboxes, assigned_scores, fg_mask):
        device = cls_preds.device
    
        assigned_labels = assigned_labels.to(device)
        assigned_bboxes = assigned_bboxes.to(device)
        assigned_scores = assigned_scores.to(device)
        fg_mask = fg_mask.to(device)
      
        num_pos = fg_mask.sum()
        if num_pos == 0:
           
            target_scores = torch.zeros_like(cls_preds[..., 0])  # [B, N]
            cls_loss = self.bce_loss(cls_preds[..., 0], target_scores).mean()
            
            return cls_loss, {
                'loss_cls': cls_loss.item(),
                'loss_bbox': 0.0,
                'loss_dfl': 0.0
            }

        B, N, C = cls_preds.shape
        target_scores = torch.zeros((B, N), device=device, dtype=cls_preds.dtype)
        
       
        if num_pos > 0:
            target_scores[fg_mask] = assigned_scores[fg_mask].squeeze(-1)
        
      
        cls_loss = self.bce_loss(cls_preds[..., 0], target_scores).mean()

       
        bbox_loss = torch.tensor(0.0, device=device)
        dfl_loss = torch.tensor(0.0, device=device)
        
        if num_pos > 0:
          
            pos_mask = fg_mask.bool()
            dfl_preds_pos = dfl_preds[pos_mask]
            assigned_bboxes_pos = assigned_bboxes[pos_mask]
            
            anchor_points_expanded = anchor_points.unsqueeze(0).expand(B, -1, -1) 
            anchor_points_pos = anchor_points_expanded[pos_mask]
            
           
            pred_bboxes_pos = self._decode_dfl_single(dfl_preds_pos, anchor_points_pos)
            
     
            bbox_loss = self._compute_iou_loss(pred_bboxes_pos, assigned_bboxes_pos)
            
           
            dfl_loss = self._compute_dfl_loss(dfl_preds_pos, assigned_bboxes_pos, anchor_points_pos)

        total_loss = self.lambda_cls * cls_loss + self.lambda_bbox * bbox_loss + self.lambda_dfl * dfl_loss

        return total_loss, {
            'loss_cls': cls_loss.item(),
            'loss_bbox': bbox_loss.item(),
            'loss_dfl': dfl_loss.item()
        }
    
    def _decode_dfl_single(self, dfl_preds, anchor_points):
        num_pos, _ = dfl_preds.shape
        dfl_preds = dfl_preds.view(num_pos, 4, self.reg_max)
        dfl_probs = torch.softmax(dfl_preds, dim=-1)
        proj = torch.arange(self.reg_max, device=dfl_preds.device, dtype=dfl_preds.dtype)
        ltrb = (dfl_probs * proj).sum(-1)
        
        x1 = anchor_points[:, 0:1] - ltrb[:, 0:1]
        y1 = anchor_points[:, 1:2] - ltrb[:, 1:2]
        x2 = anchor_points[:, 0:1] + ltrb[:, 2:3]
        y2 = anchor_points[:, 1:2] + ltrb[:, 3:4]
        
        return torch.cat([x1, y1, x2, y2], dim=-1)
    
    def _compute_iou_loss(self, pred_bboxes, target_bboxes):

        x1 = torch.max(pred_bboxes[:, 0], target_bboxes[:, 0])
        y1 = torch.max(pred_bboxes[:, 1], target_bboxes[:, 1])
        x2 = torch.min(pred_bboxes[:, 2], target_bboxes[:, 2])
        y2 = torch.min(pred_bboxes[:, 3], target_bboxes[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
       
        area_pred = (pred_bboxes[:, 2] - pred_bboxes[:, 0]) * (pred_bboxes[:, 3] - pred_bboxes[:, 1])
        area_target = (target_bboxes[:, 2] - target_bboxes[:, 0]) * (target_bboxes[:, 3] - target_bboxes[:, 1])
        
       
        union = area_pred + area_target - intersection
        
       
        iou = intersection / (union + 1e-7)
        
        
        return (1 - iou).mean()
    
    def _compute_dfl_loss(self, dfl_preds, target_bboxes, anchor_points):
       
        target_ltrb = torch.cat([
            anchor_points[:, 0:1] - target_bboxes[:, 0:1],  # left
            anchor_points[:, 1:2] - target_bboxes[:, 1:2],  # top
            target_bboxes[:, 2:3] - anchor_points[:, 0:1],  # right
            target_bboxes[:, 3:4] - anchor_points[:, 1:2],  # bottom
        ], dim=-1)
        
       
        target_ltrb = torch.clamp(target_ltrb, min=0, max=self.reg_max - 1)
        
      
        num_pos = dfl_preds.shape[0]
        dfl_preds = dfl_preds.view(num_pos, 4, self.reg_max)
        
        dfl_loss = 0
        for i in range(4):
            target_left = target_ltrb[:, i].floor().long()
            target_right = target_left + 1
            weight_left = target_right.float() - target_ltrb[:, i]
            weight_right = 1 - weight_left
            
           
            target_soft = torch.zeros_like(dfl_preds[:, i])
            target_soft.scatter_(1, target_left.unsqueeze(1), weight_left.unsqueeze(1))
            target_soft.scatter_(1, torch.clamp(target_right.unsqueeze(1), max=self.reg_max-1), weight_right.unsqueeze(1))
            
           
            dfl_loss += -(target_soft * torch.log_softmax(dfl_preds[:, i], dim=-1)).sum(dim=-1).mean()
        
        return dfl_loss / 4


def convert_to_preds(cls_preds, dfl_preds, anchor_points, reg_max=16, conf_thres=0.25, iou_thres=0.5, max_det=300):
    B, N, num_classes = cls_preds.shape
    results = []

    for b in range(B):
        probs = torch.softmax(cls_preds[b], dim=-1)
        scores, labels = probs.max(dim=-1)

        boxes = decode_dfl(dfl_preds[b].unsqueeze(0), anchor_points=anchor_points, reg_max=reg_max)[0]

        mask = scores > conf_thres
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        if len(boxes) > 0:
            keep = nms(boxes, scores, iou_thres)
            if len(keep) > max_det:
                keep = keep[:max_det]
            
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

        results.append({
            'boxes': boxes,
            'scores': scores,
            'labels': labels
        })

    return results


def convert_targets_for_map(targets):
    """Конвертация целей для вычисления mAP"""
    converted = []
    for target in targets:
        if target.numel() == 0:
            converted.append({
                'boxes': torch.empty((0, 4)),
                'labels': torch.empty((0,), dtype=torch.long)
            })
        else:
            labels = target[:, 0].long()
            xc, yc, w, h = target[:, 1], target[:, 2], target[:, 3], target[:, 4]
            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2
            boxes = torch.stack([x1, y1, x2, y2], dim=1)
            converted.append({
                'boxes': boxes,
                'labels': labels
            })
    return converted


def compute_map(results, targets):
    """Вычисление mAP"""
    if not results or not targets:
        return 0.0
    
    # Конвертация целей
    targets_converted = convert_targets_for_map(targets)
    
    # Инициализация метрики mAP
    metric = MeanAveragePrecision()
    
    # Обновление метрики
    metric.update(results, targets_converted)
    
    # Вычисление результата
    map_result = metric.compute()
    
    return map_result['map'].item()