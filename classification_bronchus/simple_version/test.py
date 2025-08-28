"""
简化版测试脚本
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from config import Config
from dataset import create_dataloaders
from model import create_model

class Evaluator:
    """评估器"""
    
    def __init__(self, model_path=None):
        Config.create_dirs()
        
        # 创建数据加载器
        _, _, self.test_loader, self.class_to_idx = create_dataloaders()
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # 加载模型
        self.model = self.load_model(model_path)
        
    def load_model(self, model_path=None):
        """加载模型"""
        if model_path is None:
            model_path = os.path.join(Config.WEIGHT_DIR, "best_model.pth")
        
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            print("请先训练模型或指定正确的模型路径")
            return None
        
        print(f"加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=Config.DEVICE)
        
        # 创建模型
        model = create_model()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"模型加载成功 (Epoch: {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']:.2f}%)")
        return model
    
    def predict(self):
        """预测测试集"""
        if self.model is None:
            return None, None
        
        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="测试")
            for inputs, targets in pbar:
                inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
                
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_targets), np.array(all_probs)
    
    def calculate_metrics(self, preds, targets):
        """计算评估指标"""
        accuracy = (preds == targets).mean() * 100
        
        # 分类报告
        class_names = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]
        report = classification_report(
            targets, preds, 
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        return accuracy, report
    
    def plot_confusion_matrix(self, targets, preds, save_path=None):
        """绘制混淆矩阵"""
        cm = confusion_matrix(targets, preds)
        class_names = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('混淆矩阵')
        plt.xlabel('预测类别')
        plt.ylabel('真实类别')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵保存至: {save_path}")
        
        plt.show()
    
    def plot_class_accuracy(self, report, save_path=None):
        """绘制各类别准确率"""
        class_names = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for class_name, metrics in report.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                class_names.append(class_name)
                precisions.append(metrics['precision'])
                recalls.append(metrics['recall'])
                f1_scores.append(metrics['f1-score'])
        
        x = np.arange(len(class_names))
        width = 0.25
        
        plt.figure(figsize=(15, 6))
        plt.bar(x - width, precisions, width, label='Precision', alpha=0.8)
        plt.bar(x, recalls, width, label='Recall', alpha=0.8)
        plt.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        plt.xlabel('类别')
        plt.ylabel('分数')
        plt.title('各类别性能指标')
        plt.xticks(x, class_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"类别准确率图保存至: {save_path}")
        
        plt.show()
    
    def evaluate(self):
        """完整评估"""
        print("开始评估...")
        
        # 预测
        preds, targets, probs = self.predict()
        if preds is None:
            return
        
        # 计算指标
        accuracy, report = self.calculate_metrics(preds, targets)
        
        # 打印结果
        print(f"\n总体准确率: {accuracy:.2f}%")
        print("\n详细分类报告:")
        print("-" * 80)
        
        # 打印每个类别的指标
        for class_name, metrics in report.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                print(f"{class_name:20s} - "
                      f"Precision: {metrics['precision']:.3f}, "
                      f"Recall: {metrics['recall']:.3f}, "
                      f"F1: {metrics['f1-score']:.3f}, "
                      f"Support: {metrics['support']}")
        
        print("-" * 80)
        print(f"{'Macro Avg':20s} - "
              f"Precision: {report['macro avg']['precision']:.3f}, "
              f"Recall: {report['macro avg']['recall']:.3f}, "
              f"F1: {report['macro avg']['f1-score']:.3f}")
        
        print(f"{'Weighted Avg':20s} - "
              f"Precision: {report['weighted avg']['precision']:.3f}, "
              f"Recall: {report['weighted avg']['recall']:.3f}, "
              f"F1: {report['weighted avg']['f1-score']:.3f}")
        
        # 绘制图表
        cm_path = os.path.join(Config.LOG_DIR, "confusion_matrix.png")
        self.plot_confusion_matrix(targets, preds, cm_path)
        
        acc_path = os.path.join(Config.LOG_DIR, "class_accuracy.png")
        self.plot_class_accuracy(report, acc_path)
        
        return accuracy, report

def main():
    """主函数"""
    try:
        evaluator = Evaluator()
        evaluator.evaluate()
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()