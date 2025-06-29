import torch
import torch.nn as nn

class YOLOv8(nn.Module):
    def __init__(self, input_size,confidence_threshold,iou_threshold, num_classes):
        super().__init__()
        pass