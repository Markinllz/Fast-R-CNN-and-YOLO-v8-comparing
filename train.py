import hydra
from omegaconf import DictConfig
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from utils.logger import logger
from models.models import YOLOv8

from dataloader.dataloader import create_loaders


class YOLOv8Trainer:
    def __init__(self, cfg : DictConfig, logger):

        self.cfg = cfg
        self.device = self._setup_device()

        self.logger = self._setup_logger()

        self._create_directories()

        self._setup_mlflow()

        #############

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        #############

        self.mAP = []
        self.train_losses = []
        self.val_losses = []
        self.bestmAP = 0.0



    def _setup_device(self):
        if self.cfg.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif self.cfg.device == "cuda" or self.cfg.device == "cpu":
            device = torch.device(self.cfg.device)
        else:
            raise ValueError(f"Unsupported device: {self.cfg.device}")
        logger.info(f"Device is {self.cfg.device}")

        return device


    def _setup_logger(self):
        return logger


    def _setup_mlflow(self):
        mlflow.set_experiment('yolo_eye_seg')
        mlflow.start_run()

        mlflow.log_params({
            "epochs": self.cfg.training.epochs,
            "lr": self.cfg.training.learning_rate,
            "batch_size": self.cfg.training.batch_size
        })


    def _setup_model(self):
        if (self.cfg.model_type == 'yolo'):
            if (self.cfg.pretrained == True):
                pass
            else:
                self.model = YOLOv8(self.cfg.input_size, self.cfg.confidence_threshold, self.cfg.iou_threshold, self.cfg.num_classes)
                self.model = self.model.to(self.device)




    def _setup_criterion(self):
        pass

    def _setup_optimizer(self):
        pass

    def _setup_scheduler(self):
        pass

    def _val_epoch(self, train_loader, epoch):
        pass

    def _train_epoch(self, val_loader, epoch):
        pass

    def _compute_losses(self):
        pass
    

    def save_checkpoint(self, epoch, val_loss, metrics, is_best=False):
        pass

    def _setup_train(self):
        self.logger.info("Начинаем настройку тренировки...")
        
        self._setup_model()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_criterion()
        
        self.logger.info("Настройка тренировки завершена")
        
    def train(self):
        pass

@hydra.main(config_path='configs', config_name = 'training/default', version_base="1.1")
def main (cfg : DictConfig):
    pass



if __name__ == "__main__":
    main()