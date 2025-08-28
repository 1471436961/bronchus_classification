"""
高效的评估器类 - 支气管分类项目
提供全面的模型评估和分析功能
"""

import os
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import label_binarize

from loguru import logger
from tqdm import tqdm

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        class_names: Optional[List[str]] = None,
        device: str = 'cuda',
        use_amp: bool = True,
        save_dir: str = '../evaluation_results'
    ):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.class_names = class_names or [f'Class_{i}' for i in range(len(test_loader.dataset.classes))]
        self.device = device
        self.use_amp = use_amp and torch.cuda.is_available()
        
        # 创建保存目录
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 存储预测结果
        self.all_predictions = []
        self.all_targets = []
        self.all_probabilities = []
        self.all_features = []  # 用于特征可视化
        
    def predict(self, return_features: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """对数据集进行预测"""
        self.model.eval()
        
        all_preds = []
        all_targets = []
        all_probs = []
        all_features = []
        
        logger.info("Starting model evaluation...")
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc="Evaluating"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)
                
                # 获取预测和概率
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_probs.append(probabilities.cpu().numpy())
                
                # 如果需要特征，提取倒数第二层的特征
                if return_features and hasattr(self.model, 'features'):
                    features = self.model.features(inputs)
                    features = torch.flatten(features, 1)
                    all_features.append(features.cpu().numpy())
        
        # 合并所有批次的结果
        self.all_predictions = np.concatenate(all_preds)
        self.all_targets = np.concatenate(all_targets)
        self.all_probabilities = np.concatenate(all_probs)
        
        if return_features and all_features:
            self.all_features = np.concatenate(all_features)
        
        logger.info(f"Evaluation completed. Total samples: {len(self.all_predictions)}")
        
        return self.all_predictions, self.all_targets, self.all_probabilities
    
    def compute_metrics(self) -> Dict[str, Any]:
        """计算各种评估指标"""
        if len(self.all_predictions) == 0:
            self.predict()
        
        metrics = {}
        
        # 基本指标
        metrics['accuracy'] = accuracy_score(self.all_targets, self.all_predictions)
        metrics['precision_macro'] = precision_score(self.all_targets, self.all_predictions, average='macro')
        metrics['precision_micro'] = precision_score(self.all_targets, self.all_predictions, average='micro')
        metrics['recall_macro'] = recall_score(self.all_targets, self.all_predictions, average='macro')
        metrics['recall_micro'] = recall_score(self.all_targets, self.all_predictions, average='micro')
        metrics['f1_macro'] = f1_score(self.all_targets, self.all_predictions, average='macro')
        metrics['f1_micro'] = f1_score(self.all_targets, self.all_predictions, average='micro')
        
        # 每个类别的指标
        class_report = classification_report(
            self.all_targets, self.all_predictions,
            target_names=self.class_names,
            output_dict=True
        )
        metrics['per_class_metrics'] = class_report
        
        # 混淆矩阵
        cm = confusion_matrix(self.all_targets, self.all_predictions)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Top-k准确率
        for k in [1, 3, 5]:
            if k <= len(self.class_names):
                top_k_acc = self._compute_top_k_accuracy(k)
                metrics[f'top_{k}_accuracy'] = top_k_acc
        
        logger.info(f"Computed metrics: Accuracy={metrics['accuracy']:.4f}, "
                   f"F1_macro={metrics['f1_macro']:.4f}")
        
        return metrics
    
    def _compute_top_k_accuracy(self, k: int) -> float:
        """计算Top-k准确率"""
        top_k_preds = np.argsort(self.all_probabilities, axis=1)[:, -k:]
        correct = 0
        for i, target in enumerate(self.all_targets):
            if target in top_k_preds[i]:
                correct += 1
        return correct / len(self.all_targets)
    
    def plot_confusion_matrix(self, normalize: bool = True, figsize: Tuple[int, int] = (12, 10)):
        """绘制混淆矩阵"""
        if len(self.all_predictions) == 0:
            self.predict()
        
        cm = confusion_matrix(self.all_targets, self.all_predictions)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm, annot=True, fmt=fmt, cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        save_path = self.save_dir / f'confusion_matrix_{"normalized" if normalize else "raw"}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {save_path}")
    
    def plot_roc_curves(self, figsize: Tuple[int, int] = (12, 8)):
        """绘制ROC曲线"""
        if len(self.all_predictions) == 0:
            self.predict()
        
        n_classes = len(self.class_names)
        
        # 二值化标签
        y_test_bin = label_binarize(self.all_targets, classes=range(n_classes))
        
        # 计算每个类别的ROC曲线
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], self.all_probabilities[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # 计算微平均ROC曲线
        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_test_bin.ravel(), self.all_probabilities.ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # 绘制ROC曲线
        plt.figure(figsize=figsize)
        
        # 绘制微平均ROC曲线
        plt.plot(
            fpr["micro"], tpr["micro"],
            label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.2f})',
            color='deeppink', linestyle=':', linewidth=4
        )
        
        # 绘制每个类别的ROC曲线
        colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
        for i, color in zip(range(min(n_classes, 10)), colors):  # 最多显示10个类别
            plt.plot(
                fpr[i], tpr[i], color=color, lw=2,
                label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.2f})'
            )
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        save_path = self.save_dir / 'roc_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curves saved to {save_path}")
        
        return roc_auc
    
    def plot_precision_recall_curves(self, figsize: Tuple[int, int] = (12, 8)):
        """绘制Precision-Recall曲线"""
        if len(self.all_predictions) == 0:
            self.predict()
        
        n_classes = len(self.class_names)
        y_test_bin = label_binarize(self.all_targets, classes=range(n_classes))
        
        plt.figure(figsize=figsize)
        
        colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
        for i, color in zip(range(min(n_classes, 10)), colors):
            precision, recall, _ = precision_recall_curve(
                y_test_bin[:, i], self.all_probabilities[:, i]
            )
            avg_precision = auc(recall, precision)
            
            plt.plot(
                recall, precision, color=color, lw=2,
                label=f'{self.class_names[i]} (AP = {avg_precision:.2f})'
            )
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        save_path = self.save_dir / 'precision_recall_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Precision-Recall curves saved to {save_path}")
    
    def analyze_misclassifications(self, top_n: int = 20) -> pd.DataFrame:
        """分析误分类样本"""
        if len(self.all_predictions) == 0:
            self.predict()
        
        # 找出误分类的样本
        misclassified_indices = np.where(self.all_predictions != self.all_targets)[0]
        
        misclassifications = []
        for idx in misclassified_indices:
            true_class = self.class_names[self.all_targets[idx]]
            pred_class = self.class_names[self.all_predictions[idx]]
            confidence = self.all_probabilities[idx, self.all_predictions[idx]]
            
            misclassifications.append({
                'sample_index': idx,
                'true_class': true_class,
                'predicted_class': pred_class,
                'confidence': confidence,
                'true_class_prob': self.all_probabilities[idx, self.all_targets[idx]]
            })
        
        # 转换为DataFrame并按置信度排序
        df = pd.DataFrame(misclassifications)
        df = df.sort_values('confidence', ascending=False).head(top_n)
        
        # 保存结果
        save_path = self.save_dir / 'misclassifications.csv'
        df.to_csv(save_path, index=False)
        
        logger.info(f"Misclassification analysis saved to {save_path}")
        logger.info(f"Total misclassified samples: {len(misclassified_indices)}")
        
        return df
    
    def generate_report(self) -> Dict[str, Any]:
        """生成完整的评估报告"""
        logger.info("Generating comprehensive evaluation report...")
        
        # 预测
        self.predict()
        
        # 计算指标
        metrics = self.compute_metrics()
        
        # 生成可视化
        self.plot_confusion_matrix(normalize=False)
        self.plot_confusion_matrix(normalize=True)
        roc_auc = self.plot_roc_curves()
        self.plot_precision_recall_curves()
        
        # 分析误分类
        misclassifications_df = self.analyze_misclassifications()
        
        # 创建报告
        report = {
            'summary': {
                'total_samples': len(self.all_predictions),
                'num_classes': len(self.class_names),
                'accuracy': metrics['accuracy'],
                'f1_macro': metrics['f1_macro'],
                'f1_micro': metrics['f1_micro'],
                'precision_macro': metrics['precision_macro'],
                'recall_macro': metrics['recall_macro']
            },
            'detailed_metrics': metrics,
            'roc_auc_scores': roc_auc,
            'misclassifications_count': len(misclassifications_df),
            'class_names': self.class_names
        }
        
        # 保存报告
        report_path = self.save_dir / 'evaluation_report.json'
        with open(report_path, 'w') as f:
            # 处理numpy类型以便JSON序列化
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                return obj
            
            json.dump(report, f, indent=2, default=convert_numpy)
        
        # 生成文本报告
        self._generate_text_report(report)
        
        logger.info(f"Evaluation report saved to {report_path}")
        logger.info(f"All evaluation results saved to {self.save_dir}")
        
        return report
    
    def _generate_text_report(self, report: Dict[str, Any]):
        """生成文本格式的报告"""
        text_report = []
        text_report.append("=" * 80)
        text_report.append("MODEL EVALUATION REPORT")
        text_report.append("=" * 80)
        text_report.append("")
        
        # 摘要
        summary = report['summary']
        text_report.append("SUMMARY")
        text_report.append("-" * 40)
        text_report.append(f"Total Samples: {summary['total_samples']}")
        text_report.append(f"Number of Classes: {summary['num_classes']}")
        text_report.append(f"Overall Accuracy: {summary['accuracy']:.4f}")
        text_report.append(f"Macro F1-Score: {summary['f1_macro']:.4f}")
        text_report.append(f"Micro F1-Score: {summary['f1_micro']:.4f}")
        text_report.append(f"Macro Precision: {summary['precision_macro']:.4f}")
        text_report.append(f"Macro Recall: {summary['recall_macro']:.4f}")
        text_report.append("")
        
        # 每个类别的详细指标
        text_report.append("PER-CLASS METRICS")
        text_report.append("-" * 40)
        per_class = report['detailed_metrics']['per_class_metrics']
        
        text_report.append(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        text_report.append("-" * 70)
        
        for class_name in self.class_names:
            if class_name in per_class:
                metrics = per_class[class_name]
                text_report.append(
                    f"{class_name:<20} "
                    f"{metrics['precision']:<10.4f} "
                    f"{metrics['recall']:<10.4f} "
                    f"{metrics['f1-score']:<10.4f} "
                    f"{metrics['support']:<10.0f}"
                )
        
        text_report.append("")
        text_report.append(f"Misclassified Samples: {report['misclassifications_count']}")
        text_report.append("")
        text_report.append("=" * 80)
        
        # 保存文本报告
        report_path = self.save_dir / 'evaluation_report.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(text_report))
        
        # 同时输出到控制台
        logger.info('\n' + '\n'.join(text_report))

def load_model_and_evaluate(
    model_path: str,
    model_class: Any,
    test_loader: DataLoader,
    config: Any,
    class_names: Optional[List[str]] = None,
    save_dir: Optional[str] = None
) -> Dict[str, Any]:
    """加载模型并进行评估的便捷函数"""
    
    # 创建模型
    model = model_class(num_classes=config.cls_num)
    
    # 处理ArcFace模型
    if hasattr(config, 'use_arc') and config.use_arc:
        from loss import ArcMarginProduct, ArcModel
        arc_head = ArcMarginProduct(
            model.classifier[2].in_features, config.cls_num,
            config.arc_config['s'], config.arc_config['m'], config.arc_config['easy_margin']
        )
        del model.classifier[2]
        model = ArcModel(model, arc_head)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'best_weight' in checkpoint:
        model.load_state_dict(checkpoint['best_weight'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # 设置保存目录
    if save_dir is None:
        model_name = Path(model_path).stem
        save_dir = f'../evaluation_results/{model_name}'
    
    # 创建评估器并运行评估
    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        class_names=class_names,
        save_dir=save_dir
    )
    
    return evaluator.generate_report()