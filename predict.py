import torch
from torchvision.ops import nms
from utils.losses import process_outputs, generate_anchors, decode_dlf

def yolo_inference(model, images, conf_thres=0.25, iou_thres=0.5, reg_max=16, num_classes=2, max_det=300, device='cpu'):

    model.eval()

    model = model.to(device)
    images = images.to(device)

    with torch.no_grad():

        preds = model(images)

        cls_preds, dfl_preds = process_outputs()