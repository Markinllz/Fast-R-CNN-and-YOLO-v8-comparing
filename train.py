import hydra
from omegaconf import DictConfig
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from utils.logger import logger
from models.models import YOLOv8
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.losses import YoloLoss
from utils.losses import generate_anchors, prepare_targets, process_outputs, assign_predictions_to_targets
from utils.losses import convert_to_preds, convert_targets_for_map, compute_map
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from dataloader.dataloader import create_loaders


class YOLOv8Trainer:
    def __init__(self, cfg : DictConfig, logger):

        self.cfg = cfg

        self.device = self._setup_device()

        self.logger = self._setup_logger()

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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device is {device}")

        return device


    def _setup_logger(self):
        return logger


    def _setup_mlflow(self):
        mlflow.set_experiment('yolo_eye_seg')
        mlflow.start_run()

        mlflow.log_params({
            "epochs": self.cfg.epochs,
            "lr": self.cfg.learning_rate,
            "batch_size": self.cfg.batch_size
        })


    def _setup_model(self):
        if (self.cfg.model_type == 'yolo'):
            if (self.cfg.pretrained == True):
                pass
            else:
                self.model = YOLOv8(self.cfg.confidence_threshold, self.cfg.iou_threshold, self.cfg.num_classes, self.cfg.reg_max)
                self.model = self.model.to(self.device)


    def _setup_criterion(self):
        self.criterion = YoloLoss(
        num_classes=self.cfg.num_classes,
        reg_max=self.cfg.reg_max,
        device=self.device,
        lambda_cls=self.cfg.loss.lambda_cls,
        lambda_bbox=self.cfg.loss.lambda_bbox,
        lambda_dfl=self.cfg.loss.lambda_dfl
        )
        self.logger.info("Criterion is ready")

    def _setup_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr = 3*1e-3)
        self.logger.info("Optimizer is ready")

    def _setup_scheduler(self):
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size= 10,gamma=0.1)
        self.logger.info("Scheduler is ready")


    @torch.no_grad()
    def _val_epoch(self, val_loader, epoch):
        running_val_loss = 0
        self.model.eval()
        total_batches = len(val_loader)
        mAP_metric = MeanAveragePrecision()


        for part, batch in enumerate(tqdm(val_loader, desc = f"Val Epoch {epoch+1}")):
            images = batch['images'].to(self.device)
            targets = batch['targets']

            preds = self.model(images)


            cls_preds, dfl_preds = process_outputs(preds)
            anchor_points, stride_tensor = generate_anchors(preds, strides=[8, 16, 32])
            pred_list = convert_to_preds(cls_preds, dfl_preds, anchor_points)
            target_list = convert_targets_for_map(targets)
            mAP_metric.update(pred_list, target_list)

            loss, loss_items = self._compute_losses(preds, targets)

            running_val_loss += loss.item()


            if part % 10 == 0:
                self.logger.info(
                f"Val Epoch {epoch} [{part}/{total_batches}] "
                f"Loss: {loss.item():.4f} | "
                f"Cls: {loss_items['loss_cls']:.4f} | "
                f"BBox: {loss_items['loss_bbox']:.4f} | "
                f"DFL: {loss_items['loss_dfl']:.4f}"
                )
        avg_loss = running_val_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        self.logger.info(f"Val Epoch {epoch} Average Loss: {avg_loss:.4f}")
        map_results = mAP_metric.compute()
        mAP = map_results['map'].item()
        return avg_loss, mAP


    def _train_epoch(self, train_loader, epoch):
        running_loss = 0
        self.model.train()
        for part, batch in enumerate(tqdm(train_loader, desc =f"Train Epoch {epoch+1}")):
            images = batch['images'].to(self.device)
            targets = batch['targets']

            self.optimizer.zero_grad()

            preds = self.model(images)


            loss, loss_items = self._compute_losses(preds, targets)

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

            if part % 10 == 0:
                self.logger.info(
                f"Train Epoch {epoch} [{part}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f} | "
                f"Cls: {loss_items['loss_cls']:.4f} | "
                f"BBox: {loss_items['loss_bbox']:.4f} | "
                f"DFL: {loss_items['loss_dfl']:.4f}"
                )

        avg_loss = running_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        self.logger.info(f"Train Epoch {epoch} Average Loss: {avg_loss:.4f}")



    def _compute_losses(self, preds, target):
        #Strides
        strides = [8,16,32]

        #Preprocessing preds:
        cls_preds, dfl_preds = process_outputs(preds=preds)


        #Get anchors
        anchor_points, stride_tensor = generate_anchors(preds=preds,strides=strides)

        #Preprocessing targets

        gt_labels, gt_bboxes, mask_gt = prepare_targets(target,self.cfg.batch_size, self.device)

        #Task Aligned Assigner

        assigned_labels, assigned_bboxes, assigned_scores, fg_mask, target_gt_idx = assign_predictions_to_targets(cls_preds,dfl_preds,anchor_points,gt_labels,gt_bboxes,mask_gt, device=self.device)
        #Loss

        loss, loss_items = self.criterion(
        cls_preds, dfl_preds, assigned_labels, assigned_bboxes, assigned_scores, fg_mask
        )

        return loss, loss_items
    

    def save_checkpoint(self, epoch, val_loss, metrics, is_best=False):
        save_dir = Path('./checkpoints')
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch' : epoch,
            'model_state_dict': self.model.state_dict(),
            'model_state_optimizer': self.optimizer.state_dict(),
            'model_state_scheduler' : self.scheduler.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'bestmAP': self.bestmAP
        }

        checkpoint_path = save_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint сохранён: {checkpoint_path}")

        if is_best:
            best_path = save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            mlflow.log_artifact(str(best_path))
            self.logger.info(f"Лучший чекпоинт обновлён: {best_path}")

    def _setup_train(self):
        self.logger.info("Начинаем настройку тренировки...")
        
        self._setup_model()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_criterion()
        
        self.logger.info("Настройка тренировки завершена")
        
    def train(self):
        self._setup_train()
        train_loader, val_loader = create_loaders(self.cfg)
        num_epochs = self.cfg.epochs

        for epoch in range(num_epochs):
            

            self.logger.info(f"=== Эпоха {epoch+1}/{num_epochs} ===")

            self._train_epoch(train_loader,epoch)

            val_loss , mAP = self._val_epoch(val_loader, epoch)


            mlflow.log_metric("train_loss", self.train_losses[-1], step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_mAP", mAP, step=epoch)


            is_best = mAP > self.bestmAP if self.bestmAP > 0 else True

            if is_best:
                self.bestmAP = mAP

            self.save_checkpoint(epoch, val_loss, metrics = {'mAP' : mAP}, is_best=is_best)

            if self.scheduler is not None:
                self.scheduler.step()



@hydra.main(config_path='configs', config_name = 'config', version_base="1.1")
def main (cfg : DictConfig):
    trainer = YOLOv8Trainer(cfg, logger)
    trainer.train()
    mlflow.end_run()


if __name__ == "__main__":
    main()
    
