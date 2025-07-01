try:
    from .dataset import EyeCataractDataset
except ImportError:
    from dataset import EyeCataractDataset

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
utils_path = os.path.join(project_root, "utils")
if utils_path not in sys.path:
    sys.path.append(utils_path)
from logger import logger


def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    
    targets = []
    
    for batch_idx, item in enumerate(batch):
        labels = item['labels'][0]
        
       
        if len(labels) == 5:
           
            target = torch.tensor([labels], dtype=torch.float32)
        else:
           
            target = torch.zeros((0, 5), dtype=torch.float32)
            
        targets.append(target)
    
    return {
        'images': images,
        'targets': targets
    }



def create_loaders(cfg):
    logger.info("🔧 Создание DataLoader'ов...")
    
    
    logger.info("📂 Загрузка val датасета...")
    logger.info(f"   📊 Batch size: {cfg.batch_size}")
    logger.info(f"   🖼️  Image size: {cfg.input_size}")
    logger.info(f"   👥 Workers: {cfg.num_workers}")
    logger.info(f"   🔀 Shuffle: {cfg.shuffle}")
    logger.info(f"   ✂️  Drop last: {cfg.drop_last}")

    train_dataset = EyeCataractDataset(
        images_dir=f"{cfg.train_path}/images",
        labels_dir=f"{cfg.train_path}/labels",
        model_type=cfg.model_type,
        img_size=cfg.input_size
    )

    val_dataset = EyeCataractDataset(
        images_dir=f"{cfg.val_path}/images", 
        labels_dir=f"{cfg.val_path}/labels",
        model_type=cfg.model_type,
        img_size=cfg.input_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_fn,
        drop_last=cfg.drop_last
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_fn,
        drop_last=False
    )
        
    train_batches = len(train_loader)
    val_batches = len(val_loader)
    
    logger.info("✅ DataLoader'ы созданы успешно!")
    logger.info(f"   📊 Train: {train_batches} батчей ({len(train_dataset)} изображений)")
    logger.info(f"   📊 Val: {val_batches} батчей ({len(val_dataset)} изображений)")

    return train_loader, val_loader





