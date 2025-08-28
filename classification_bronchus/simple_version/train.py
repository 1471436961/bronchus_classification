"""
简化版训练脚本
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import Config
from dataset import create_dataloaders
from model import create_model

class Trainer:
    """训练器"""
    
    def __init__(self):
        Config.create_dirs()
        Config.print_config()
        
        # 创建数据加载器
        self.train_loader, self.val_loader, self.test_loader, self.class_to_idx = create_dataloaders()
        
        # 创建模型
        self.model = create_model()
        
        # 创建优化器和调度器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=Config.NUM_EPOCHS,
            eta_min=1e-6
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 记录训练历史
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        
        self.best_val_acc = 0.0
        self.best_model_path = os.path.join(Config.WEIGHT_DIR, "best_model.pth")
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="训练")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """验证"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="验证")
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                pbar.set_postfix({
                    'Loss': f'{running_loss/(batch_idx+1):.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def save_model(self, epoch, val_acc, is_best=False):
        """保存模型"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'class_to_idx': self.class_to_idx,
            'config': {
                'model_name': Config.MODEL_NAME,
                'num_classes': Config.NUM_CLASSES,
                'image_size': Config.IMAGE_SIZE
            }
        }
        
        if is_best:
            torch.save(checkpoint, self.best_model_path)
            print(f"保存最佳模型: {self.best_model_path}")
        
        # 保存最新模型
        latest_path = os.path.join(Config.WEIGHT_DIR, "latest_model.pth")
        torch.save(checkpoint, latest_path)
    
    def plot_training_history(self):
        """绘制训练历史"""
        plt.figure(figsize=(12, 4))
        
        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='训练损失')
        plt.plot(self.val_losses, label='验证损失')
        plt.title('损失曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='训练准确率')
        plt.plot(self.val_accs, label='验证准确率')
        plt.title('准确率曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(Config.LOG_DIR, "training_history.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"训练历史图保存至: {plot_path}")
    
    def train(self):
        """开始训练"""
        print("开始训练...")
        start_time = time.time()
        
        for epoch in range(Config.NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
            print("-" * 50)
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc = self.validate()
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # 打印结果
            print(f"训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.2f}%")
            print(f"验证 - 损失: {val_loss:.4f}, 准确率: {val_acc:.2f}%")
            print(f"学习率: {current_lr:.6f}")
            
            # 保存模型
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                print(f"新的最佳验证准确率: {val_acc:.2f}%")
            
            self.save_model(epoch, val_acc, is_best)
            
            # 早停检查（简化版本，可以根据需要扩展）
            if epoch > 20 and val_acc < max(self.val_accs[-10:]) - 5:
                print("验证准确率长时间未提升，考虑早停...")
        
        # 训练完成
        total_time = time.time() - start_time
        print(f"\n训练完成! 总用时: {total_time/3600:.2f} 小时")
        print(f"最佳验证准确率: {self.best_val_acc:.2f}%")
        
        # 绘制训练历史
        self.plot_training_history()

def main():
    """主函数"""
    try:
        trainer = Trainer()
        trainer.train()
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()