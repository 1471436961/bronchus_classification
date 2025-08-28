#!/usr/bin/env python3
"""
高效预处理的训练脚本
整合了所有数据预处理功能
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# 导入工具模块
from utils import (
    setup_training_logger, get_logger, init_default_logger,
    get_device, move_to_device, clear_cuda_cache,
    validate_config, ValidationError,
    handle_exceptions, profile_function,
    MemoryManager, global_monitor
)

init_default_logger()
logger = get_logger(__name__)
import matplotlib.pyplot as plt
import seaborn as sns

# 导入核心模块
from config import Config, ConfigTemplates
from data_process import (
    DatasetSplitter, 
    MedicalImageAugmentation, 
    Dataset, 
    CustomDataLoader,
    calculate_dataset_statistics
)


class OptimizedTrainer:
    """高效的训练器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.system.device)
        
        # 设置随机种子
        self._set_seed(config.system.seed)
        
        # 创建输出目录
        self.output_dir = Path(config.system.output_dir) / f"train_{time.strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        config.save_config(str(self.output_dir / "config.json"))
        
        # 初始化日志
        self.writer = SummaryWriter(str(self.output_dir / "tensorboard"))
        
        # 训练历史
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }
        
        # 最佳指标
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        print(f"🚀 训练器初始化完成，输出目录: {self.output_dir}")
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
        if self.config.system.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def prepare_data(self) -> Tuple[torch.utils.data.DataLoader, ...]:
        """准备数据"""
        print("📊 准备数据...")
        
        # 检查数据是否存在
        if not os.path.exists(self.config.data.train_path):
            print("❌ 训练数据不存在，尝试自动划分数据集...")
            self._split_dataset()
        
        # 计算数据集统计信息（如果需要）
        if self.config.data.use_custom_normalization:
            print("📈 计算数据集统计信息...")
            stats = calculate_dataset_statistics(self.config.data.train_path)
            # 更新标准化参数
            # 这里可以动态更新增强策略中的标准化参数
        
        # 创建数据增强策略
        input_size = self.config.data.input_size
        
        if self.config.data.augmentation_level == "light":
            train_transform = MedicalImageAugmentation.get_light_augmentation(input_size)
        elif self.config.data.augmentation_level == "heavy":
            train_transform = MedicalImageAugmentation.get_heavy_augmentation(input_size)
        else:  # medium
            train_transform = MedicalImageAugmentation.get_medium_augmentation(input_size)
        
        val_transform = MedicalImageAugmentation.get_test_transform(input_size)
        
        # 创建数据集
        train_dataset = Dataset(
            data_dir=self.config.data.train_path,
            transform=train_transform,
            cache_images=self.config.data.cache_images
        )
        
        val_dataset = Dataset(
            data_dir=self.config.data.val_path,
            transform=val_transform,
            class_to_idx=train_dataset.class_to_idx,  # 保持一致的类别映射
            cache_images=self.config.data.cache_images
        )
        
        # 保存类别映射
        with open(self.output_dir / "class_to_idx.json", 'w') as f:
            json.dump(train_dataset.class_to_idx, f, indent=2)
        
        # 创建数据加载器
        train_loader = CustomDataLoader.create_dataloader(
            dataset=train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            persistent_workers=self.config.data.persistent_workers,
            prefetch_factor=self.config.data.prefetch_factor,
            use_weighted_sampler=self.config.data.use_weighted_sampler
        )
        
        val_loader = CustomDataLoader.create_dataloader(
            dataset=val_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            persistent_workers=self.config.data.persistent_workers,
            prefetch_factor=self.config.data.prefetch_factor,
            use_weighted_sampler=False
        )
        
        print(f"✅ 数据准备完成:")
        print(f"   训练集: {len(train_dataset)} 样本")
        print(f"   验证集: {len(val_dataset)} 样本")
        print(f"   类别数: {len(train_dataset.classes)}")
        print(f"   批次大小: {self.config.data.batch_size}")
        print(f"   数据增强: {self.config.data.augmentation_level}")
        
        return train_loader, val_loader, train_dataset.classes
    
    def _split_dataset(self):
        """自动划分数据集"""
        # 假设原始数据在 data_root 目录下
        org_path = self.config.data.data_root.replace("/data_split", "")
        if os.path.exists(org_path):
            splitter = DatasetSplitter(random_state=self.config.system.seed)
            splitter.split_data_stratified(
                org_path=org_path,
                split_path=self.config.data.data_root,
                train_ratio=self.config.data.train_ratio,
                val_ratio=self.config.data.val_ratio,
                test_ratio=self.config.data.test_ratio,
                min_samples_per_class=self.config.data.min_samples_per_class
            )
        else:
            raise FileNotFoundError(f"找不到原始数据目录: {org_path}")
    
    def create_model(self) -> nn.Module:
        """创建模型"""
        print(f"🏗️ 创建模型: {self.config.model.model_name}")
        
        model = self.config.model.create_model()
        model = model.to(self.device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ 模型创建完成:")
        print(f"   总参数: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        print(f"   输入尺寸: {self.config.data.input_size}")
        
        return model
    
    def create_optimizer_and_scheduler(self, model: nn.Module) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
        """创建训练器和学习率调度器"""
        # 创建训练器
        if self.config.training.optimizer.lower() == "adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer.lower() == "adamw":
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer.lower() == "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.config.training.learning_rate,
                momentum=self.config.training.momentum,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"不支持的训练器: {self.config.training.optimizer}")
        
        # 创建学习率调度器
        if self.config.training.scheduler.lower() == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.training.epochs
            )
        elif self.config.training.scheduler.lower() == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=self.config.training.step_size,
                gamma=self.config.training.gamma
            )
        elif self.config.training.scheduler.lower() == "multistep":
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.config.training.milestones,
                gamma=self.config.training.gamma
            )
        elif self.config.training.scheduler.lower() == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=self.config.training.patience,
                factor=self.config.training.gamma
            )
        else:
            scheduler = None
        
        print(f"✅ 训练器: {self.config.training.optimizer}")
        print(f"✅ 学习率调度器: {self.config.training.scheduler}")
        
        return optimizer, scheduler
    
    def create_loss_function(self, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
        """创建损失函数"""
        if self.config.training.loss_function == "cross_entropy":
            if self.config.training.label_smoothing > 0:
                criterion = nn.CrossEntropyLoss(
                    weight=class_weights,
                    label_smoothing=self.config.training.label_smoothing
                )
            else:
                criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            # 可以添加其他损失函数
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        return criterion.to(self.device)
    
    def train_epoch(
        self, 
        model: nn.Module, 
        train_loader: torch.utils.data.DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        epoch: int = 0
    ) -> Tuple[float, float]:
        """训练一个epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.training.epochs}")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # 前向传播
            if self.config.training.use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                # 反向传播
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.config.training.accumulate_grad_batches == 0:
                    if self.config.training.gradient_clip_val > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.training.gradient_clip_val)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                loss.backward()
                
                if (batch_idx + 1) % self.config.training.accumulate_grad_batches == 0:
                    if self.config.training.gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.training.gradient_clip_val)
                    
                    optimizer.step()
                    optimizer.zero_grad()
            
            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 更新进度条
            if batch_idx % self.config.system.log_every_n_steps == 0:
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%',
                    'LR': f'{current_lr:.2e}'
                })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(
        self, 
        model: nn.Module, 
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """验证一个epoch"""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validating"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config.__dict__
        }
        
        # 保存最新检查点
        torch.save(checkpoint, self.output_dir / "latest_checkpoint.pth")
        
        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, self.output_dir / "best_model.pth")
            torch.save(model.state_dict(), self.output_dir / f"{self.config.model.model_name}_best.pth")
    
    def plot_training_history(self):
        """绘制训练历史"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # 损失曲线
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Train Acc')
        ax2.plot(epochs, self.history['val_acc'], 'r-', label='Val Acc')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # 学习率曲线
        ax3.plot(epochs, self.history['lr'], 'g-', label='Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True)
        
        # 验证准确率详细视图
        ax4.plot(epochs, self.history['val_acc'], 'r-', linewidth=2)
        ax4.axhline(y=max(self.history['val_acc']), color='g', linestyle='--', 
                   label=f'Best: {max(self.history["val_acc"]):.2f}%')
        ax4.set_title('Validation Accuracy (Detailed)')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy (%)')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_history.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def train(self):
        """主训练循环"""
        print("🎯 开始训练...")
        
        # 准备数据
        train_loader, val_loader, classes = self.prepare_data()
        
        # 创建模型
        model = self.create_model()
        
        # 创建训练器和调度器
        optimizer, scheduler = self.create_optimizer_and_scheduler(model)
        
        # 创建损失函数
        class_weights = None
        if self.config.data.use_class_weights:
            # 这里可以从训练数据集计算类别权重
            pass
        criterion = self.create_loss_function(class_weights)
        
        # 混合精度训练
        scaler = torch.cuda.amp.GradScaler() if self.config.training.use_amp else None
        
        # 训练循环
        start_time = time.time()
        
        for epoch in range(self.config.training.epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(
                model, train_loader, optimizer, criterion, scaler, epoch
            )
            
            # 验证
            val_loss, val_acc = self.validate_epoch(model, val_loader, criterion)
            
            # 更新学习率
            current_lr = optimizer.param_groups[0]['lr']
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # TensorBoard日志
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Val', val_acc, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # 检查是否为最佳模型
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # 保存检查点
            self.save_checkpoint(model, optimizer, epoch, is_best)
            
            # 打印epoch结果
            print(f"Epoch {epoch+1}/{self.config.training.epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  LR: {current_lr:.2e}, Best Val Acc: {self.best_val_acc:.2f}%")
            
            # 早停检查
            if (self.config.training.early_stopping and 
                self.patience_counter >= self.config.training.early_stopping_patience):
                print(f"早停触发！连续 {self.patience_counter} 个epoch无改善")
                break
        
        # 训练完成
        total_time = time.time() - start_time
        print(f"\n🎉 训练完成！")
        print(f"总训练时间: {total_time/3600:.2f} 小时")
        print(f"最佳验证准确率: {self.best_val_acc:.2f}%")
        
        # 绘制训练历史
        self.plot_training_history()
        
        # 保存训练历史
        with open(self.output_dir / "training_history.json", 'w') as f:
            json.dump(self.history, f, indent=2)
        
        self.writer.close()
        
        return model


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="高效预处理训练脚本")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--template", type=str, choices=["fast", "high_quality", "small_dataset", "large_model"],
                       help="使用预定义配置模板")
    parser.add_argument("--model", type=str, default="efficientnet_b0", help="模型名称")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=16, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--augmentation", type=str, choices=["light", "medium", "heavy"], 
                       default="medium", help="数据增强强度")
    
    args = parser.parse_args()
    
    # 创建配置
    if args.config:
        config = Config.load_config(args.config)
    elif args.template:
        if args.template == "fast":
            config = ConfigTemplates.get_fast_training_config()
        elif args.template == "high_quality":
            config = ConfigTemplates.get_high_quality_config()
        elif args.template == "small_dataset":
            config = ConfigTemplates.get_small_dataset_config()
        elif args.template == "large_model":
            config = ConfigTemplates.get_large_model_config()
    else:
        config = Config()
    
    # 应用命令行参数
    if args.model:
        config.model.model_name = args.model
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.augmentation:
        config.data.augmentation_level = args.augmentation
    
    # 创建训练器并开始训练
    trainer = OptimizedTrainer(config)
    model = trainer.train()
    
    print("✅ 训练完成！")


if __name__ == "__main__":
    main()