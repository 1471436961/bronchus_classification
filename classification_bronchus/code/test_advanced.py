#!/usr/bin/env python3
"""
高级评估脚本 - 支气管分类项目
提供全面的模型评估功能和深度分析
"""

import os
import sys
import argparse
import json
import shutil
import datetime
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    balanced_accuracy_score, precision_recall_fscore_support,
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc
)
from loguru import logger
import pandas as pd

from config import Config, ConfigTemplates
from data_process import Dataset, CustomDataLoader, MedicalImageAugmentation

# 防止OMP错误
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class AdvancedEvaluator:
    """高级模型评估器 - 提供全面的模型评估和分析功能"""
    
    def __init__(self, config: Config, save_dir: str = "./evaluation_results"):
        self.config = config
        self.device = torch.device(config.system.device)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.images_dir = self.save_dir / "error_images"
        self.plots_dir = self.save_dir / "plots"
        self.reports_dir = self.save_dir / "reports"
        
        for dir_path in [self.images_dir, self.plots_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志系统"""
        cur_time = datetime.datetime.now().strftime('test_%Y%m%d_%H%M%S')
        log_path = self.save_dir / f"{cur_time}.log"
        logger.add(str(log_path), rotation="10 MB", retention="7 days")
        logger.info(f"Advanced evaluator initialized at {cur_time}")
        
    def load_model(self, model_path: str) -> nn.Module:
        """加载模型"""
        logger.info(f"Loading model from: {model_path}")
        print(f"🔄 加载模型: {model_path}")
        
        # 创建模型
        model = self.config.model.create_model()
        
        # 加载权重
        if model_path.endswith('.pth'):
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 处理不同的保存格式
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    if 'best_acc' in checkpoint:
                        logger.info(f"Model best accuracy: {checkpoint['best_acc']}")
                elif 'best_weight' in checkpoint:
                    # 旧版本格式
                    weights = checkpoint['best_weight']
                    model.load_state_dict(weights)
                    if 'best_acc' in checkpoint:
                        logger.info(f"Model best accuracy: {checkpoint['best_acc']}")
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        logger.info("Model loaded successfully")
        print(f"✅ 模型加载完成")
        return model
    
    def create_test_loader(self) -> Tuple[DataLoader, List[str], List[str]]:
        """创建数据加载器"""
        logger.info("Creating data loader...")
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
        
        # 获取图像路径（用于错误分析）
        image_paths = []
        if hasattr(test_dataset, 'samples'):
            image_paths = [sample[0] for sample in test_dataset.samples]
        elif hasattr(test_dataset, 'imgs'):
            image_paths = [img[0] for img in test_dataset.imgs]
        
        logger.info(f"Test dataset created: {len(test_dataset)} samples, {len(class_names)} classes")
        print(f"✅ 数据加载器创建完成:")
        print(f"   样本数: {len(test_dataset)}")
        print(f"   类别数: {len(class_names)}")
        print(f"   批次数: {len(test_loader)}")
        
        return test_loader, class_names, image_paths
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader, 
                      class_names: List[str], image_paths: List[str]) -> Dict:
        """全面评估模型"""
        logger.info("Starting comprehensive model evaluation...")
        print("🧪 开始全面模型评估...")
        
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
        
        # 转换为numpy数组
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        
        # 计算基础指标
        accuracy = accuracy_score(all_targets, all_predictions)
        balanced_acc = balanced_accuracy_score(all_targets, all_predictions)
        
        # 计算macro和weighted平均指标
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='macro'
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted'
        )
        
        # 计算每个类别的指标
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            all_targets, all_predictions, average=None
        )
        
        # 分类报告
        class_report = classification_report(
            all_targets, all_predictions, 
            target_names=class_names, 
            output_dict=True
        )
        
        # 混淆矩阵
        cm = confusion_matrix(all_targets, all_predictions)
        
        # 计算每个类别的准确率（旧版本的方法）
        accuracy_per_class = []
        for i in range(len(class_names)):
            if i < len(cm):
                TP = cm[i, i]
                total = np.sum(cm[i, :])
                class_accuracy = TP / total if total != 0 else 0
                accuracy_per_class.append(class_accuracy)
            else:
                accuracy_per_class.append(0.0)
        
        # 计算ROC AUC（如果是二分类或多分类）
        roc_auc_scores = {}
        if len(class_names) == 2:
            # 二分类
            roc_auc_scores['binary'] = roc_auc_score(all_targets, all_probabilities[:, 1])
        elif len(class_names) > 2:
            # 多分类 - 计算每个类别的OvR AUC
            try:
                roc_auc_scores['macro'] = roc_auc_score(all_targets, all_probabilities, 
                                                      multi_class='ovr', average='macro')
                roc_auc_scores['weighted'] = roc_auc_score(all_targets, all_probabilities, 
                                                         multi_class='ovr', average='weighted')
            except ValueError as e:
                logger.warning(f"Could not compute ROC AUC: {e}")
                roc_auc_scores = {}
        
        # 找出预测错误的样本
        error_indices = np.where(all_predictions != all_targets)[0]
        error_samples = []
        for idx in error_indices:
            if idx < len(image_paths):
                error_samples.append({
                    'image_path': image_paths[idx],
                    'true_label': int(all_targets[idx]),
                    'predicted_label': int(all_predictions[idx]),
                    'true_class': class_names[all_targets[idx]],
                    'predicted_class': class_names[all_predictions[idx]],
                    'confidence': float(all_probabilities[idx][all_predictions[idx]])
                })
        
        results = {
            # 基础指标
            'accuracy': float(accuracy),
            'balanced_accuracy': float(balanced_acc),
            
            # Macro平均指标
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_score_macro': float(f1_macro),
            
            # Weighted平均指标
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_score_weighted': float(f1_weighted),
            
            # 每个类别的指标
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_score_per_class': f1_per_class.tolist(),
            'accuracy_per_class': accuracy_per_class,
            'support_per_class': support_per_class.tolist(),
            
            # ROC AUC
            'roc_auc_scores': roc_auc_scores,
            
            # 详细数据
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'predictions': all_predictions.tolist(),
            'targets': all_targets.tolist(),
            'probabilities': all_probabilities.tolist(),
            'class_names': class_names,
            'error_samples': error_samples,
            
            # 统计信息
            'total_samples': len(all_targets),
            'error_count': len(error_samples),
            'error_rate': len(error_samples) / len(all_targets)
        }
        
        # 打印结果摘要
        print(f"✅ 评估完成:")
        print(f"   总样本数: {len(all_targets)}")
        print(f"   准确率: {accuracy:.4f}")
        print(f"   平衡准确率: {balanced_acc:.4f}")
        print(f"   精确率(macro): {precision_macro:.4f}")
        print(f"   召回率(macro): {recall_macro:.4f}")
        print(f"   F1分数(macro): {f1_macro:.4f}")
        print(f"   错误样本数: {len(error_samples)}")
        print(f"   错误率: {len(error_samples) / len(all_targets):.4f}")
        
        # 打印每个类别的准确率
        print(f"\n📊 各类别准确率:")
        for i, (class_name, acc) in enumerate(zip(class_names, accuracy_per_class)):
            print(f"   {class_name}: {acc:.4f}")
        
        logger.info(f"Evaluation completed: accuracy={accuracy:.4f}, balanced_accuracy={balanced_acc:.4f}")
        
        return results
    
    def save_error_images(self, error_samples: List[Dict], max_errors: int = 100):
        """保存预测错误的图像"""
        if not error_samples:
            print("✅ 没有预测错误的样本")
            return
        
        print(f"💾 保存预测错误的图像 (最多{max_errors}张)...")
        logger.info(f"Saving error images: {len(error_samples)} errors found")
        
        saved_count = 0
        for error in error_samples[:max_errors]:
            try:
                src_path = error['image_path']
                if os.path.exists(src_path):
                    # 创建新的文件名
                    base_name = os.path.basename(src_path)
                    name, ext = os.path.splitext(base_name)
                    new_name = f"{name}_true{error['true_label']}_pred{error['predicted_label']}_conf{error['confidence']:.3f}{ext}"
                    
                    dst_path = self.images_dir / new_name
                    shutil.copy2(src_path, dst_path)
                    
                    # 记录日志
                    logger.info(f"Error image saved: {src_path} -> true:{error['true_class']}, pred:{error['predicted_class']}, conf:{error['confidence']:.3f}")
                    saved_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to save error image {error['image_path']}: {e}")
        
        print(f"✅ 已保存 {saved_count} 张错误预测图像到: {self.images_dir}")
    
    def save_results(self, results: Dict, model_name: str = "model"):
        """保存评估结果"""
        print("💾 保存评估结果...")
        logger.info("Saving evaluation results...")
        
        # 保存数值结果
        numeric_results = {
            'model_name': model_name,
            'evaluation_time': datetime.datetime.now().isoformat(),
            'total_samples': results['total_samples'],
            'error_count': results['error_count'],
            'error_rate': results['error_rate'],
            
            # 基础指标
            'accuracy': results['accuracy'],
            'balanced_accuracy': results['balanced_accuracy'],
            
            # Macro平均指标
            'precision_macro': results['precision_macro'],
            'recall_macro': results['recall_macro'],
            'f1_score_macro': results['f1_score_macro'],
            
            # Weighted平均指标
            'precision_weighted': results['precision_weighted'],
            'recall_weighted': results['recall_weighted'],
            'f1_score_weighted': results['f1_score_weighted'],
            
            # 每个类别的指标
            'per_class_metrics': {
                'class_names': results['class_names'],
                'precision': results['precision_per_class'],
                'recall': results['recall_per_class'],
                'f1_score': results['f1_score_per_class'],
                'accuracy': results['accuracy_per_class'],
                'support': results['support_per_class']
            },
            
            # ROC AUC
            'roc_auc_scores': results['roc_auc_scores'],
            
            # 分类报告
            'classification_report': results['classification_report']
        }
        
        # 保存JSON结果
        with open(self.reports_dir / f"{model_name}_results.json", 'w', encoding='utf-8') as f:
            json.dump(numeric_results, f, indent=2, ensure_ascii=False)
        
        # 保存详细的错误分析
        if results['error_samples']:
            with open(self.reports_dir / f"{model_name}_error_analysis.json", 'w', encoding='utf-8') as f:
                json.dump(results['error_samples'], f, indent=2, ensure_ascii=False)
        
        # 保存预测错误的图像
        self.save_error_images(results['error_samples'])
        
        # 生成可视化图表
        self.plot_confusion_matrix(np.array(results['confusion_matrix']), results['class_names'], model_name)
        self.plot_classification_report(results['classification_report'], model_name)
        self.plot_per_class_metrics(results, model_name)
        self.plot_error_analysis(results, model_name)
        
        print(f"✅ 结果已保存到: {self.save_dir}")
        logger.info(f"Results saved to: {self.save_dir}")
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], model_name: str):
        """绘制混淆矩阵"""
        plt.figure(figsize=(12, 10))
        
        # 计算百分比
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # 创建热力图
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title(f'Confusion Matrix (%) - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(self.plots_dir / f"{model_name}_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 同时保存数值版本
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title(f'Confusion Matrix (Count) - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(self.plots_dir / f"{model_name}_confusion_matrix_count.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_classification_report(self, report: Dict, model_name: str):
        """绘制分类报告"""
        classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
        metrics = ['precision', 'recall', 'f1-score']
        
        # 创建DataFrame
        data = []
        for cls in classes:
            for metric in metrics:
                data.append([cls, metric, report[cls][metric]])
        
        df = pd.DataFrame(data, columns=['Class', 'Metric', 'Score'])
        
        # 绘制条形图
        plt.figure(figsize=(15, 8))
        
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
        
        plt.savefig(self.plots_dir / f"{model_name}_classification_report.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_per_class_metrics(self, results: Dict, model_name: str):
        """绘制每个类别的详细指标"""
        class_names = results['class_names']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Per-Class Metrics - {model_name}', fontsize=16)
        
        # 精确率
        axes[0, 0].bar(class_names, results['precision_per_class'])
        axes[0, 0].set_title('Precision per Class')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 召回率
        axes[0, 1].bar(class_names, results['recall_per_class'])
        axes[0, 1].set_title('Recall per Class')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # F1分数
        axes[1, 0].bar(class_names, results['f1_score_per_class'])
        axes[1, 0].set_title('F1-Score per Class')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 准确率
        axes[1, 1].bar(class_names, results['accuracy_per_class'])
        axes[1, 1].set_title('Accuracy per Class')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f"{model_name}_per_class_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_error_analysis(self, results: Dict, model_name: str):
        """绘制错误分析图"""
        if not results['error_samples']:
            return
        
        # 错误分布分析
        error_by_true_class = {}
        error_by_pred_class = {}
        
        for error in results['error_samples']:
            true_class = error['true_class']
            pred_class = error['predicted_class']
            
            error_by_true_class[true_class] = error_by_true_class.get(true_class, 0) + 1
            error_by_pred_class[pred_class] = error_by_pred_class.get(pred_class, 0) + 1
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Error Analysis - {model_name}', fontsize=16)
        
        # 按真实类别的错误分布
        if error_by_true_class:
            axes[0].bar(error_by_true_class.keys(), error_by_true_class.values())
            axes[0].set_title('Errors by True Class')
            axes[0].set_ylabel('Error Count')
            axes[0].tick_params(axis='x', rotation=45)
        
        # 按预测类别的错误分布
        if error_by_pred_class:
            axes[1].bar(error_by_pred_class.keys(), error_by_pred_class.values())
            axes[1].set_title('Errors by Predicted Class')
            axes[1].set_ylabel('Error Count')
            axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f"{model_name}_error_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="高级支气管分类模型评估脚本")
    parser.add_argument("--model-path", type=str, required=True, help="模型权重路径")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--template", type=str, help="配置模板")
    parser.add_argument("--model", type=str, default="efficientnet_b0", help="模型架构")
    parser.add_argument("--test-data-path", type=str, help="数据路径")
    parser.add_argument("--batch-size", type=int, default=16, help="批次大小")
    parser.add_argument("--save-dir", type=str, default="./evaluation_results_enhanced", help="结果保存目录")
    parser.add_argument("--device", type=str, default="auto", help="设备 (cuda/cpu/auto)")
    parser.add_argument("--max-error-images", type=int, default=100, help="最大保存错误图像数量")
    
    args = parser.parse_args()
    
    # 创建配置
    if args.config:
        config = Config.load_config(args.config)
    elif args.template:
        config = ConfigTemplates.get_template(args.template)
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
    
    print(f"🚀 开始高级模型评估")
    print(f"模型路径: {args.model_path}")
    print(f"模型架构: {config.model.model_name}")
    print(f"数据路径: {config.data.test_path}")
    print(f"设备: {config.system.device}")
    print(f"结果保存: {args.save_dir}")
    
    # 创建高级评估器
    evaluator = AdvancedEvaluator(config, args.save_dir)
    
    try:
        # 加载模型
        model = evaluator.load_model(args.model_path)
        
        # 创建数据加载器
        test_loader, class_names, image_paths = evaluator.create_test_loader()
        
        # 评估模型
        results = evaluator.evaluate_model(model, test_loader, class_names, image_paths)
        
        # 保存结果
        model_name = Path(args.model_path).stem
        evaluator.save_results(results, model_name)
        
        print(f"\n🎉 高级评估完成！")
        print(f"结果已保存到: {args.save_dir}")
        print(f"  - 报告: {args.save_dir}/reports/")
        print(f"  - 图表: {args.save_dir}/plots/")
        print(f"  - 错误图像: {args.save_dir}/error_images/")
        
    except Exception as e:
        print(f"❌ 评估过程中出错: {e}")
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()