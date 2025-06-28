try:
    from .dataset import EyeCataractDataset  # Для импорта из пакета
except ImportError:
    from dataset import EyeCataractDataset   # Для прямого запуска

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
    
    labels = [item['labels'][0] for item in batch]
    
    targets = []
    
    for batch_idx, label in enumerate(labels):
        target = [batch_idx] + label 
        targets.append(target)
    
    targets = torch.tensor(targets, dtype=torch.float32)
    
    return {
        'images': images,
        'targets': targets
    }



def create_loaders(cfg):
    logger.info("🔧 Создание DataLoader'ов...")
    
    logger.info(f"   📊 Batch size: {cfg.data.batch_size}")
    logger.info(f"   🖼️  Image size: {cfg.model.input_size}")
    logger.info(f"   👥 Workers: {cfg.data.num_workers}")
    logger.info(f"   🔀 Shuffle: {cfg.data.shuffle}")
    logger.info(f"   ✂️  Drop last: {cfg.data.drop_last}")


    train_dataset = EyeCataractDataset(
        images_dir=f"{cfg.data.train_path}/images",
        labels_dir=f"{cfg.data.train_path}/labels",
        model_type=cfg.data.model_type,
        img_size=cfg.model.input_size
    )
    logger.info(f"   ✅ Train: {len(train_dataset)} изображений")
    
    logger.info("📂 Загрузка val датасета...")
    val_dataset = EyeCataractDataset(
        images_dir=f"{cfg.data.val_path}/images", 
        labels_dir=f"{cfg.data.val_path}/labels",
        model_type=cfg.data.model_type,
        img_size=cfg.model.input_size
    )
    logger.info(f"   ✅ Val: {len(val_dataset)} изображений")
    
    logger.info("🔧 Создание train DataLoader...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate_fn,
        drop_last=cfg.data.drop_last
    )
    
    logger.info("🔧 Создание val DataLoader...")
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate_fn,
        drop_last=False
    )
    
    train_batches = len(train_loader)
    val_batches = len(val_loader)
    
    logger.info("✅ DataLoader'ы созданы успешно!")
    logger.info(f"   📊 Train: {train_batches} батчей ({len(train_dataset)} изображений)")
    logger.info(f"   📊 Val: {val_batches} батчей ({len(val_dataset)} изображений)")

    return train_loader, val_loader





