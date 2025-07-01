import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
from pathlib import Path
from utils.logger import logger
from ultralytics import YOLO
import yaml
import os


class YOLOv8Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = logger
        self.model = None
 
        self._setup_mlflow()

    def _setup_mlflow(self):

        mlflow.set_tracking_uri("file:./mlruns")
        self.logger.info("✅ MLFlow настроен для YOLOv8")

    def _create_dataset_yaml(self):
       
        
  
        dataset_root = "/Users/bagdasaryanproduction/Eye Segmentation/Dataset"
       
        train_images = Path(dataset_root) / "train" / "images"
        val_images = Path(dataset_root) / "val" / "images"
        
        if not train_images.exists():
            raise FileNotFoundError(f"Train images не найдены: {train_images}")
        if not val_images.exists():
            raise FileNotFoundError(f"Val images не найдены: {val_images}")
        
       
        class_names = ["normal_eye", "cataract_eye"] 
        
        dataset_yaml = {
            'path': dataset_root,
            'train': 'train/images', 
            'val': 'val/images',
            'nc': len(class_names), 
            'names': class_names 
        }
        
        yaml_path = Path('dataset.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False, allow_unicode=True)
        
        self.logger.info(f"✅ Создан dataset.yaml:")
        self.logger.info(f"   📁 Путь: {dataset_yaml['path']}")
        self.logger.info(f"   🚂 Train: {train_images} ({'существует' if train_images.exists() else 'НЕ НАЙДЕН'})")
        self.logger.info(f"   🔍 Val: {val_images} ({'существует' if val_images.exists() else 'НЕ НАЙДЕН'})")
        self.logger.info(f"   🏷️  Классы: {class_names}")
        self.logger.info(f"   📊 Количество классов: {len(class_names)}")
        
        return str(yaml_path)

    def setup_model(self):
        
        if self.cfg.pretrained:
          
            self.model = YOLO('yolov8n.pt')
            self.logger.info("✅ Загружена предобученная YOLOv8n модель")
        else:
            self.model = YOLO('yolov8n.yaml')
            self.logger.info("✅ Создана YOLOv8n модель с нуля")

    def train(self):
       
        self.logger.info("🚀 Начинаем обучение YOLOv8 для детекции глаз...")
        
      
        dataset_yaml = self._create_dataset_yaml()
       
        self.setup_model()
        
        
        results = self.model.train(
            data=dataset_yaml,
            epochs=self.cfg.epochs,
            imgsz=self.cfg.input_size,
            batch=self.cfg.batch_size,
            lr0=self.cfg.learning_rate,
            device='mps' if self.cfg.device == 'auto' else self.cfg.device,
            workers=self.cfg.num_workers,
            project='runs/detect',
            name='eye_detection',
            save=True,
            save_period=self.cfg.checkpoint.save_every,
            patience=self.cfg.early_stopping.patience if self.cfg.early_stopping.enabled else 0,
            verbose=True,
            
            conf=self.cfg.confidence_threshold,
            iou=self.cfg.iou_threshold,
           
            hsv_h=0.015, 
            hsv_s=0.7,    
            hsv_v=0.4,   
            degrees=0.0,   
            translate=0.1,
            scale=0.5,     
            shear=0.0,     
            perspective=0.0, 
            flipud=0.0,    
            fliplr=0.5,   
            mosaic=1.0,    
            mixup=0.0,   
        )
        
        self.logger.info("✅ Обучение завершено!")
        return results

    def validate(self):
       
        if self.model is None:
            self.logger.error("Модель не загружена!")
            return None
            
        self.logger.info("🔍 Запуск валидации...")
        
        dataset_yaml = self._create_dataset_yaml()
        
        results = self.model.val(
            data=dataset_yaml,
            imgsz=self.cfg.input_size,
            batch=self.cfg.batch_size,
            device='mps' if self.cfg.device == 'auto' else self.cfg.device,
            verbose=True,
            save_json=True,
            save_hybrid=True,
        )
        
        self.logger.info("✅ Валидация завершена!")
        
        return results

    def save_model(self, path=None):
      
        if self.model is None:
            self.logger.error("Модель не загружена!")
            return
            
        if path is None:
            path = "checkpoints/yolo_eye_detection_best.pt"
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        self.logger.info(f"✅ Модель сохранена: {path}")


@hydra.main(config_path='configs', config_name='config', version_base="1.1")
def main(cfg: DictConfig):
    logger.info("🎯 Запуск обучения YOLOv8 для детекции глаз")
    logger.info(f"📊 Конфигурация:")
    logger.info(f"   📈 Эпохи: {cfg.epochs}")
    logger.info(f"   📦 Batch size: {cfg.batch_size}")
    logger.info(f"   🖼️  Размер изображения: {cfg.input_size}")
    
    
    trainer = YOLOv8Trainer(cfg)
    
    try:
      
        train_results = trainer.train()
        
        
        val_results = trainer.validate()
        
        trainer.save_model()
        
        logger.info("🎉 Процесс обучения успешно завершен!")
        
        
        if train_results:
            logger.info("📊 Результаты обучения:")
            
            best_model_path = Path("runs/detect/eye_detection/weights/best.pt")
            if best_model_path.exists():
                logger.info(f"   🏆 Лучшая модель: {best_model_path}")
            
    except Exception as e:
        logger.error(f"Ошибка во время обучения: {e}")
        raise


if __name__ == "__main__":
    main()