#!/usr/bin/env python3
"""
高效的评估脚本 - 支气管分类项目
使用高效的配置和数据处理系统
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve

from config import Config, ConfigTemplates
from data_process import Dataset, CustomDataLoader, MedicalImageAugmentation

# 防止OMP错误
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class SimpleEvaluator:
    """高效的模型评估器"""
    
    def __init__(self, config: Config, save_dir: str = "./evaluation_results"):
        self.config = config
        self.device = torch.device(config.system.device)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def load_model(self, model_path: str) -> nn.Module:
        """加载模型"""
        print(f"🔄 加载模型: {model_path}")
        
        # 创建模型
        model = self.config.model.create_model()
        
        # 加载权重
        if model_path.endswith('.pth'):
            if 'checkpoint' in model_path or 'best_model' in model_path:
                # 完整检查点
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                # 只有模型权重
                model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        model = model.to(self.device)
        model.eval()
        
        print(f"✅ 模型加载完成")
        return model
    
    def create_test_loader(self) -> Tuple[DataLoader, List[str]]:
        """创建数据加载器"""
        print(f"📊 创建数据加载器...")
        
        # 创建变换
        test_transform = MedicalImageAugmentation.get_test_transform(self.config.data.input_size)
        
        # 创建数据集
        test_dataset = Dataset(
            data_dir=self.config.data.test_path,
            transform=test_transform,
            cache_images=False
        )
        
        # 创建数据加载器
        test_loader = CustomDataLoader.create_dataloader(
            dataset=test_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            use_weighted_sampler=False
        )
        
        class_names = test_dataset.classes
        
        print(f"✅ 数据加载器创建完成:")
        print(f"   样本数: {len(test_dataset)}")
        print(f"   类别数: {len(class_names)}")
        print(f"   批次数: {len(test_loader)}")
        
        return test_loader, class_names
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader, class_names: List[str]) -> Dict:
        """评估模型"""
        print("🧪 开始模型评估...")
        
        model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc="评估中"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='weighted')
        
        # 分类报告
        class_report = classification_report(
            all_targets, all_predictions, 
            target_names=class_names, 
            output_dict=True
        )
        
        # 混淆矩阵
        cm = confusion_matrix(all_targets, all_predictions)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': np.array(all_probabilities),
            'class_names': class_names
        }
        
        print(f"✅ 评估完成:")
        print(f"   准确率: {accuracy:.4f}")
        print(f"   精确率: {precision:.4f}")
        print(f"   召回率: {recall:.4f}")
        print(f"   F1分数: {f1:.4f}")
        
        return results
    
    def save_results(self, results: Dict, model_name: str = "model"):
        """保存评估结果"""
        print("💾 保存评估结果...")
        
        # 保存数值结果
        numeric_results = {
            'accuracy': float(results['accuracy']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1_score': float(results['f1_score']),
            'classification_report': results['classification_report']
        }
        
        with open(self.save_dir / f"{model_name}_results.json", 'w', encoding='utf-8') as f:
            json.dump(numeric_results, f, indent=2, ensure_ascii=False)
        
        # 绘制混淆矩阵
        self.plot_confusion_matrix(results['confusion_matrix'], results['class_names'], model_name)
        
        # 绘制分类报告
        self.plot_classification_report(results['classification_report'], model_name)
        
        print(f"✅ 结果已保存到: {self.save_dir}")
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], model_name: str):
        """绘制混淆矩阵"""
        plt.figure(figsize=(12, 10))
        
        # 计算百分比
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # 创建热力图
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(self.save_dir / f"{model_name}_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_classification_report(self, report: Dict, model_name: str):
        """绘制分类报告"""
        # 提取每个类别的指标
        classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
        metrics = ['precision', 'recall', 'f1-score']
        
        data = []
        for cls in classes:
            for metric in metrics:
                data.append([cls, metric, report[cls][metric]])
        
        # 创建DataFrame用于绘图
        import pandas as pd
        df = pd.DataFrame(data, columns=['Class', 'Metric', 'Score'])
        
        # 绘制条形图
        plt.figure(figsize=(15, 8))
        
        # 按类别分组绘制
        x_pos = np.arange(len(classes))
        width = 0.25
        
        precision_scores = [report[cls]['precision'] for cls in classes]
        recall_scores = [report[cls]['recall'] for cls in classes]
        f1_scores = [report[cls]['f1-score'] for cls in classes]
        
        plt.bar(x_pos - width, precision_scores, width, label='Precision', alpha=0.8)
        plt.bar(x_pos, recall_scores, width, label='Recall', alpha=0.8)
        plt.bar(x_pos + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        plt.xlabel('Classes')
        plt.ylabel('Score')
        plt.title(f'Classification Report - {model_name}')
        plt.xticks(x_pos, classes, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.save_dir / f"{model_name}_classification_report.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="支气管分类模型评估脚本")
    parser.add_argument("--model-path", type=str, required=True, help="模型权重路径")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--model", type=str, default="efficientnet_b0", help="模型架构")
    parser.add_argument("--test-data-path", type=str, help="数据路径")
    parser.add_argument("--batch-size", type=int, default=16, help="批次大小")
    parser.add_argument("--save-dir", type=str, default="./evaluation_results", help="结果保存目录")
    parser.add_argument("--device", type=str, default="auto", help="设备 (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    # 创建配置
    if args.config:
        config = Config.load_config(args.config)
    else:
        config = Config()
    
    # 应用命令行参数
    if args.model:
        config.model.model_name = args.model
    if args.test_data_path:
        config.data.test_path = args.test_data_path
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.device != "auto":
        config.system.device = args.device
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"❌ 模型文件不存在: {args.model_path}")
        return
    
    # 检查数据是否存在
    if not os.path.exists(config.data.test_path):
        print(f"❌ 数据不存在: {config.data.test_path}")
        return
    
    print(f"🚀 开始模型评估")
    print(f"模型路径: {args.model_path}")
    print(f"模型架构: {config.model.model_name}")
    print(f"数据路径: {config.data.test_path}")
    print(f"设备: {config.system.device}")
    
    # 创建评估器
    evaluator = SimpleEvaluator(config, args.save_dir)
    
    try:
        # 加载模型
        model = evaluator.load_model(args.model_path)
        
        # 创建数据加载器
        test_loader, class_names = evaluator.create_test_loader()
        
        # 评估模型
        results = evaluator.evaluate_model(model, test_loader, class_names)
        
        # 保存结果
        model_name = Path(args.model_path).stem
        evaluator.save_results(results, model_name)
        
        print(f"\n🎉 评估完成！")
        print(f"结果已保存到: {args.save_dir}")
        
    except Exception as e:
        print(f"❌ 评估过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()