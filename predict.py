import hydra
from omegaconf import DictConfig
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from utils.logger import logger


@hydra.main(config_path='configs', config_name='config', version_base="1.1")
def main(cfg: DictConfig):
    logger.info("🔍 Запуск детекции глаз с YOLOv8")
   
    project_root = Path(__file__).parent.resolve()

    model_paths = [
    project_root / "outputs/2025-07-01/10-04-33/runs/detect/eye_detection/weights/best.pt",
    project_root / "checkpoints/yolo_eye_detection_best.pt",
    project_root / "yolov8n.pt"
    ]
    
    model_path = None
    for path in model_paths:
        if Path(path).exists():
            model_path = path
            break
    
    if model_path is None:
        logger.error("❌ Не найдена обученная модель!")
        return
    
   
    model = YOLO(model_path)
    logger.info(f"✅ Модель загружена: {model_path}")
    
   
    test_path = Path(cfg.test_path)
    if not test_path.exists():
        test_path = Path(cfg.val_path)
        logger.info(f"📁 Используем валидационные данные: {test_path}")
    
    
    results_dir = Path("results/detections")
    results_dir.mkdir(parents=True, exist_ok=True)
    
   
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(test_path.glob(f"*{ext}"))
        image_files.extend(test_path.glob(f"**/*{ext}"))
    
    logger.info(f"📊 Найдено {len(image_files)} изображений для детекции")
    
    if len(image_files) == 0:
        logger.warning("⚠️ Изображения не найдены!")
        return
    

    
    for i, image_path in enumerate(image_files[:20]):
        logger.info(f"🖼️  Обработка {i+1}/{min(20, len(image_files))}: {image_path.name}")
        
        results = model(
            source=str(image_path),
            imgsz=cfg.input_size,
            conf=cfg.confidence_threshold,
            iou=cfg.iou_threshold,
            save=True,
            save_txt=True,
            save_conf=True,
            project=str(results_dir),
            name=f"detection_{i+1:03d}",
            exist_ok=True,
            line_width=2,
            show_labels=True,
            show_conf=True
        )
        
    return results


if __name__ == "__main__":
    main()