"""
高效的训练器类 - 支气管分类项目
提供模块化、高效的训练和验证功能
"""

import os
import copy
import time
import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from tqdm import tqdm

from torchmetrics import (
    F1Score, Accuracy, Recall, Precision, 
    ConfusionMatrix, MetricCollection
)

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience: int = 7, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_weights = copy.deepcopy(model.state_dict())
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0
        return False

class MetricsTracker:
    """指标跟踪器"""
    def __init__(self, num_classes: int, device: str):
        self.device = device
        self.metrics = MetricCollection({
            'accuracy': Accuracy(task='multiclass', num_classes=num_classes),
            'f1_macro': F1Score(task='multiclass', num_classes=num_classes, average='macro'),
            'f1_micro': F1Score(task='multiclass', num_classes=num_classes, average='micro'),
            'recall': Recall(task='multiclass', num_classes=num_classes, average='macro'),
            'precision': Precision(task='multiclass', num_classes=num_classes, average='macro'),
        }).to(device)
        
        self.confusion_matrix = ConfusionMatrix(
            task='multiclass', num_classes=num_classes
        ).to(device)
        
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """更新指标"""
        self.metrics.update(preds, targets)
        self.confusion_matrix.update(preds, targets)
        
    def compute(self) -> Dict[str, float]:
        """计算指标"""
        metrics = self.metrics.compute()
        # 转换为Python标量
        return {k: v.item() if hasattr(v, 'item') else v for k, v in metrics.items()}
        
    def reset(self):
        """重置指标"""
        self.metrics.reset()
        self.confusion_matrix.reset()
        
    def get_confusion_matrix(self) -> torch.Tensor:
        """获取混淆矩阵"""
        return self.confusion_matrix.compute()

class Trainer:
    """高效的训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any] = None,
        config: Optional[Any] = None,
        device: str = 'cuda',
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        save_dir: str = '../weight',
        log_interval: int = 50,
        use_early_stopping: bool = True,
        patience: int = 10
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.use_amp = use_amp and torch.cuda.is_available()
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        
        # 创建保存目录
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化AMP
        self.scaler = GradScaler() if self.use_amp else None
        
        # 初始化指标跟踪器
        num_classes = config.cls_num if config else 1000
        self.train_metrics = MetricsTracker(num_classes, device)
        self.val_metrics = MetricsTracker(num_classes, device)
        
        # 早停机制
        self.early_stopping = EarlyStopping(patience=patience) if use_early_stopping else None
        
        # 训练历史
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': [],
            'lr': []
        }
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.save_dir / 'tensorboard')
        
        # 最佳模型跟踪
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.best_metrics = {}
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        # 进度条
        pbar = tqdm(
            enumerate(self.train_loader), 
            total=num_batches,
            desc=f'Epoch {epoch+1} [Train]',
            leave=False
        )
        
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # 前向传播
            if self.use_amp:
                with autocast():
                    if hasattr(self.config, 'use_arc') and self.config.use_arc:
                        outputs = self.model(inputs, targets)
                    else:
                        outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets) / self.gradient_accumulation_steps
            else:
                if hasattr(self.config, 'use_arc') and self.config.use_arc:
                    outputs = self.model(inputs, targets)
                else:
                    outputs = self.model(inputs)
                loss = self.criterion(outputs, targets) / self.gradient_accumulation_steps
            
            # 反向传播
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    # 梯度裁剪
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # 更新指标
            with torch.no_grad():
                preds = torch.argmax(outputs, dim=1)
                self.train_metrics.update(preds, targets)
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            
            # 更新进度条
            if batch_idx % self.log_interval == 0:
                current_metrics = self.train_metrics.compute()
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{current_metrics["accuracy"]:.4f}'
                })
        
        # 计算epoch指标
        epoch_metrics = self.train_metrics.compute()
        epoch_metrics['loss'] = total_loss / num_batches
        
        return epoch_metrics
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        pbar = tqdm(
            self.val_loader,
            desc=f'Epoch {epoch+1} [Val]',
            leave=False
        )
        
        with torch.no_grad():
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                if self.use_amp:
                    with autocast():
                        if hasattr(self.config, 'use_arc') and self.config.use_arc:
                            outputs = self.model(inputs, targets, training=False)
                        else:
                            outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                else:
                    if hasattr(self.config, 'use_arc') and self.config.use_arc:
                        outputs = self.model(inputs, targets, training=False)
                    else:
                        outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                preds = torch.argmax(outputs, dim=1)
                self.val_metrics.update(preds, targets)
                total_loss += loss.item()
                
                # 更新进度条
                current_metrics = self.val_metrics.compute()
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{current_metrics["accuracy"]:.4f}'
                })
        
        # 计算epoch指标
        epoch_metrics = self.val_metrics.compute()
        epoch_metrics['loss'] = total_loss / num_batches
        
        return epoch_metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'history': self.history,
            'config': self.config.__dict__ if self.config else None,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'best_metrics': self.best_metrics
        }
        
        # 保存最新检查点
        torch.save(checkpoint, self.save_dir / 'latest_checkpoint.pth')
        
        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_model.pth')
            logger.info(f'New best model saved with val_acc: {metrics["accuracy"]:.4f}')
    
    def log_metrics(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """记录指标"""
        # 更新历史
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['train_acc'].append(train_metrics['accuracy'])
        self.history['val_acc'].append(val_metrics['accuracy'])
        self.history['train_f1'].append(train_metrics['f1_macro'])
        self.history['val_f1'].append(val_metrics['f1_macro'])
        
        if self.scheduler:
            self.history['lr'].append(self.scheduler.get_last_lr()[0])
        
        # TensorBoard日志
        self.writer.add_scalars('Loss', {
            'train': train_metrics['loss'],
            'val': val_metrics['loss']
        }, epoch)
        
        self.writer.add_scalars('Accuracy', {
            'train': train_metrics['accuracy'],
            'val': val_metrics['accuracy']
        }, epoch)
        
        self.writer.add_scalars('F1_Score', {
            'train': train_metrics['f1_macro'],
            'val': val_metrics['f1_macro']
        }, epoch)
        
        if self.scheduler:
            self.writer.add_scalar('Learning_Rate', self.scheduler.get_last_lr()[0], epoch)
        
        # 控制台日志
        logger.info(
            f'Epoch {epoch+1:3d} | '
            f'Train Loss: {train_metrics["loss"]:.4f} | '
            f'Train Acc: {train_metrics["accuracy"]:.4f} | '
            f'Val Loss: {val_metrics["loss"]:.4f} | '
            f'Val Acc: {val_metrics["accuracy"]:.4f} | '
            f'Val F1: {val_metrics["f1_macro"]:.4f}'
        )
    
    def train(self, num_epochs: int, resume_from: Optional[str] = None) -> Dict[str, List[float]]:
        """主训练循环"""
        start_epoch = 0
        
        # 恢复训练
        if resume_from and os.path.exists(resume_from):
            checkpoint = torch.load(resume_from)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler and checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            self.history = checkpoint['history']
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch = checkpoint['best_epoch']
            self.best_metrics = checkpoint['best_metrics']
            logger.info(f'Resumed training from epoch {start_epoch}')
        
        logger.info(f'Starting training for {num_epochs} epochs')
        logger.info(f'Device: {self.device}')
        logger.info(f'Mixed Precision: {self.use_amp}')
        logger.info(f'Gradient Accumulation Steps: {self.gradient_accumulation_steps}')
        
        start_time = time.time()
        
        try:
            for epoch in range(start_epoch, num_epochs):
                epoch_start_time = time.time()
                
                # 训练
                train_metrics = self.train_epoch(epoch)
                
                # 验证
                val_metrics = self.validate_epoch(epoch)
                
                # 学习率调度
                if self.scheduler:
                    self.scheduler.step()
                
                # 记录指标
                self.log_metrics(epoch, train_metrics, val_metrics)
                
                # 检查是否是最佳模型
                is_best = val_metrics['accuracy'] > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_metrics['accuracy']
                    self.best_epoch = epoch
                    self.best_metrics = val_metrics.copy()
                
                # 保存检查点
                self.save_checkpoint(epoch, val_metrics, is_best)
                
                # 早停检查
                if self.early_stopping:
                    if self.early_stopping(val_metrics['accuracy'], self.model):
                        logger.info(f'Early stopping triggered at epoch {epoch+1}')
                        break
                
                epoch_time = time.time() - epoch_start_time
                logger.info(f'Epoch {epoch+1} completed in {epoch_time:.2f}s')
        
        except KeyboardInterrupt:
            logger.info('Training interrupted by user')
        
        except Exception as e:
            logger.error(f'Training failed with error: {e}')
            raise
        
        finally:
            total_time = time.time() - start_time
            logger.info(f'Training completed in {total_time:.2f}s')
            logger.info(f'Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch+1}')
            
            # 关闭TensorBoard writer
            self.writer.close()
            
            # 保存训练历史图表
            self.plot_training_history()
        
        return self.history
    
    def plot_training_history(self):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 准确率曲线
        axes[0, 1].plot(self.history['train_acc'], label='Train Acc')
        axes[0, 1].plot(self.history['val_acc'], label='Val Acc')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1分数曲线
        axes[1, 0].plot(self.history['train_f1'], label='Train F1')
        axes[1, 0].plot(self.history['val_f1'], label='Val F1')
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 学习率曲线
        if self.history['lr']:
            axes[1, 1].plot(self.history['lr'], label='Learning Rate')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f'Training history plot saved to {self.save_dir / "training_history.png"}')