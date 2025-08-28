"""
数据预处理模块
包含数据加载、数据增强、内存管理等功能
"""

import os
import random
import shutil
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import Counter
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns


class DatasetSplitter:
    """数据集划分器，支持分层采样"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
    def split_data_stratified(
        self, 
        org_path: str, 
        split_path: str, 
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        min_samples_per_class: int = 10
    ) -> Dict[str, int]:
        """
        分层采样划分数据集，确保类别平衡
        
        Args:
            org_path: 原始数据路径
            split_path: 输出路径
            train_ratio: 训练集比例
            val_ratio: 验证集比例  
            test_ratio: 验证集比例
            min_samples_per_class: 每个类别最少样本数
            
        Returns:
            统计信息字典
        """
        print(f"{'='*20} 开始分层数据集划分 {'='*20}")
        
        # 验证比例
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
        
        # 收集所有样本信息
        all_samples = []
        class_counts = Counter()
        
        for class_name in os.listdir(org_path):
            class_path = os.path.join(org_path, class_name)
            if not os.path.isdir(class_path):
                continue
                
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            if len(images) < min_samples_per_class:
                warnings.warn(f"类别 {class_name} 只有 {len(images)} 个样本，少于最小要求 {min_samples_per_class}")
                
            for img in images:
                all_samples.append((os.path.join(class_path, img), class_name))
                class_counts[class_name] += 1
        
        # 提取路径和标签
        X = [sample[0] for sample in all_samples]
        y = [sample[1] for sample in all_samples]
        
        # 第一次划分：分离出验证集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_ratio, 
            stratify=y, random_state=self.random_state
        )
        
        # 第二次划分：从剩余数据中分离训练集和验证集
        val_size_adjusted = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            stratify=y_temp, random_state=self.random_state
        )
        
        # 创建目录并复制文件
        splits = {
            'train': (X_train, y_train),
            'val': (X_val, y_val), 
            'test': (X_test, y_test)
        }
        
        stats = {}
        for split_name, (X_split, y_split) in splits.items():
            split_dir = os.path.join(split_path, split_name)
            os.makedirs(split_dir, exist_ok=True)
            
            split_counts = Counter(y_split)
            stats[split_name] = dict(split_counts)
            
            for src_path, class_name in zip(X_split, y_split):
                dst_dir = os.path.join(split_dir, class_name)
                os.makedirs(dst_dir, exist_ok=True)
                
                dst_path = os.path.join(dst_dir, os.path.basename(src_path))
                shutil.copy2(src_path, dst_path)
        
        # 保存统计信息
        self._save_split_stats(split_path, stats, class_counts)
        self._plot_split_distribution(split_path, stats)
        
        print(f"{'='*20} 数据集划分完成 {'='*20}")
        return stats
    
    def _save_split_stats(self, split_path: str, stats: Dict, original_counts: Counter):
        """保存划分统计信息"""
        stats_file = os.path.join(split_path, 'split_statistics.json')
        
        full_stats = {
            'original_distribution': dict(original_counts),
            'split_distribution': stats,
            'total_samples': {
                'original': sum(original_counts.values()),
                'train': sum(stats['train'].values()),
                'val': sum(stats['val'].values()),
                'test': sum(stats['test'].values())
            }
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(full_stats, f, indent=2, ensure_ascii=False)
            
        print(f"统计信息已保存到: {stats_file}")
    
    def _plot_split_distribution(self, split_path: str, stats: Dict):
        """绘制数据分布图"""
        try:
            # 准备数据
            classes = list(stats['train'].keys())
            train_counts = [stats['train'].get(cls, 0) for cls in classes]
            val_counts = [stats['val'].get(cls, 0) for cls in classes]
            test_counts = [stats['test'].get(cls, 0) for cls in classes]
            
            # 创建图表
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # 堆叠柱状图
            x = np.arange(len(classes))
            width = 0.8
            
            ax1.bar(x, train_counts, width, label='Train', alpha=0.8)
            ax1.bar(x, val_counts, width, bottom=train_counts, label='Val', alpha=0.8)
            ax1.bar(x, test_counts, width, 
                   bottom=np.array(train_counts) + np.array(val_counts), 
                   label='Test', alpha=0.8)
            
            ax1.set_xlabel('Classes')
            ax1.set_ylabel('Sample Count')
            ax1.set_title('Dataset Split Distribution by Class')
            ax1.set_xticks(x)
            ax1.set_xticklabels(classes, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 比例饼图
            total_train = sum(train_counts)
            total_val = sum(val_counts)
            total_test = sum(test_counts)
            
            sizes = [total_train, total_val, total_test]
            labels = ['Train', 'Val', 'Test']
            colors = ['#ff9999', '#66b3ff', '#99ff99']
            
            ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Overall Dataset Split Ratio')
            
            plt.tight_layout()
            plt.savefig(os.path.join(split_path, 'split_distribution.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"分布图已保存到: {os.path.join(split_path, 'split_distribution.png')}")
            
        except Exception as e:
            print(f"绘制分布图时出错: {e}")


class MedicalImageAugmentation:
    """医学图像专用数据增强"""
    
    @staticmethod
    def get_light_augmentation(input_size: Tuple[int, int]) -> A.Compose:
        """轻度数据增强，适合高质量数据"""
        return A.Compose([
            A.Resize(input_size[0], input_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    @staticmethod
    def get_medium_augmentation(input_size: Tuple[int, int]) -> A.Compose:
        """中等强度数据增强"""
        return A.Compose([
            A.Resize(input_size[0], input_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.15, scale_limit=0.15, rotate_limit=25, p=0.6
            ),
            A.OneOf([
                A.ElasticTransform(p=0.3),
                A.GridDistortion(p=0.3),
                A.OpticalDistortion(p=0.3),
            ], p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=0.6
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.MotionBlur(blur_limit=7, p=0.3),
            ], p=0.4),
            A.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.4
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    @staticmethod
    def get_heavy_augmentation(input_size: Tuple[int, int]) -> A.Compose:
        """强数据增强，适合数据不足的情况"""
        return A.Compose([
            A.Resize(input_size[0], input_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.4),
            A.RandomRotate90(p=0.4),
            A.Transpose(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.2, scale_limit=0.2, rotate_limit=35, p=0.7
            ),
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, p=0.4),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.4),
                A.OpticalDistortion(distort_limit=0.3, shift_limit=0.3, p=0.4),
            ], p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.4, contrast_limit=0.4, p=0.7
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10, 80), p=0.4),
                A.GaussianBlur(blur_limit=(3, 9), p=0.4),
                A.MotionBlur(blur_limit=9, p=0.4),
                A.MedianBlur(blur_limit=7, p=0.3),
            ], p=0.6),
            A.HueSaturationValue(
                hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=0.5
            ),
            A.OneOf([
                A.CLAHE(clip_limit=2, p=0.3),
                A.Sharpen(p=0.3),
                A.Emboss(p=0.3),
            ], p=0.4),
            A.CoarseDropout(
                max_holes=8, max_height=32, max_width=32, 
                min_holes=1, min_height=8, min_width=8, p=0.3
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    @staticmethod
    def get_test_transform(input_size: Tuple[int, int]) -> A.Compose:
        """推理时的变换（无增强）"""
        return A.Compose([
            A.Resize(input_size[0], input_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


class Dataset(Dataset):
    """高效的数据集类，支持缓存和多种增强策略"""
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[A.Compose] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        cache_images: bool = False,
        image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.cache_images = cache_images
        self.image_extensions = image_extensions
        
        # 构建样本列表
        self.samples, self.class_to_idx = self._make_dataset(class_to_idx)
        self.classes = list(self.class_to_idx.keys())
        
        # 图像缓存
        self._image_cache = {} if cache_images else None
        
        print(f"数据集加载完成: {len(self.samples)} 个样本, {len(self.classes)} 个类别")
        if cache_images:
            print("启用图像缓存模式")
    
    def _make_dataset(self, class_to_idx: Optional[Dict[str, int]]) -> Tuple[List[Tuple[str, int]], Dict[str, int]]:
        """构建数据集样本列表"""
        if class_to_idx is None:
            # 自动构建类别索引
            classes = sorted([d for d in os.listdir(self.data_dir) 
                            if os.path.isdir(os.path.join(self.data_dir, d))])
            class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        samples = []
        for class_name, class_idx in class_to_idx.items():
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(self.image_extensions):
                    path = os.path.join(class_dir, filename)
                    samples.append((path, class_idx))
        
        return samples, class_to_idx
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, target = self.samples[index]
        
        # 从缓存或磁盘加载图像
        if self._image_cache is not None and path in self._image_cache:
            image = self._image_cache[path]
        else:
            image = self._load_image(path)
            if self._image_cache is not None:
                self._image_cache[path] = image
        
        # 应用变换
        if self.transform is not None:
            if isinstance(image, np.ndarray):
                transformed = self.transform(image=image)
                image = transformed['image']
            else:
                # 如果是PIL图像，转换为numpy数组
                image_np = np.array(image)
                transformed = self.transform(image=image_np)
                image = transformed['image']
        else:
            # 默认转换为tensor
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            else:
                image = transforms.ToTensor()(image)
        
        return image, target
    
    def _load_image(self, path: str) -> np.ndarray:
        """加载图像，返回RGB格式的numpy数组"""
        try:
            # 使用OpenCV加载（BGR格式）
            image = cv2.imread(path)
            if image is None:
                raise ValueError(f"无法加载图像: {path}")
            
            # 转换为RGB格式
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
            
        except Exception as e:
            print(f"加载图像失败 {path}: {e}")
            # 返回黑色图像作为fallback
            return np.zeros((224, 224, 3), dtype=np.uint8)
    
    def get_class_weights(self) -> torch.Tensor:
        """计算类别权重，用于处理不平衡数据"""
        class_counts = Counter([target for _, target in self.samples])
        total_samples = len(self.samples)
        num_classes = len(self.class_to_idx)
        
        weights = torch.zeros(num_classes)
        for class_idx, count in class_counts.items():
            weights[class_idx] = total_samples / (num_classes * count)
        
        return weights


class CustomDataLoader:
    """高效的数据加载器工厂"""
    
    @staticmethod
    def create_dataloader(
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: Optional[int] = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        use_weighted_sampler: bool = False
    ) -> DataLoader:
        """
        创建高效的数据加载器
        
        Args:
            dataset: 数据集
            batch_size: 批次大小
            shuffle: 是否打乱
            num_workers: 工作进程数（None表示自动检测）
            pin_memory: 是否使用固定内存
            persistent_workers: 是否使用持久化工作进程
            prefetch_factor: 预取因子
            use_weighted_sampler: 是否使用加权采样器
        """
        if num_workers is None:
            num_workers = min(8, os.cpu_count() or 1)
        
        # 如果使用加权采样器，则不能同时shuffle
        sampler = None
        if use_weighted_sampler and hasattr(dataset, 'get_class_weights'):
            class_weights = dataset.get_class_weights()
            sample_weights = [class_weights[target] for _, target in dataset.samples]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(dataset),
                replacement=True
            )
            shuffle = False
        
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            persistent_workers=persistent_workers and num_workers > 0,
            prefetch_factor=prefetch_factor if num_workers > 0 else 2,
            drop_last=True if shuffle else False
        )


def calculate_dataset_statistics(data_dir: str, sample_size: int = 1000) -> Dict[str, float]:
    """计算数据集的统计信息（均值和标准差）"""
    print(f"计算数据集统计信息，采样 {sample_size} 张图像...")
    
    # 收集所有图像路径
    image_paths = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(root, file))
    
    # 随机采样
    if len(image_paths) > sample_size:
        image_paths = random.sample(image_paths, sample_size)
    
    # 计算统计信息
    pixel_values = []
    for path in image_paths:
        try:
            image = cv2.imread(path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pixel_values.append(image.reshape(-1, 3))
        except Exception as e:
            print(f"跳过图像 {path}: {e}")
    
    if not pixel_values:
        print("警告: 没有成功加载任何图像，使用ImageNet统计值")
        return {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    
    # 合并所有像素值
    all_pixels = np.vstack(pixel_values).astype(np.float32) / 255.0
    
    # 计算均值和标准差
    mean = np.mean(all_pixels, axis=0).tolist()
    std = np.std(all_pixels, axis=0).tolist()
    
    stats = {'mean': mean, 'std': std}
    print(f"数据集统计信息: mean={mean}, std={std}")
    
    return stats


# 使用示例和演示函数
def run_preprocessing():
    """演示数据预处理流程"""
    print("🚀 数据预处理演示")
    
    # 1. 数据集划分
    splitter = DatasetSplitter(random_state=42)
    
    # 2. 创建增强策略
    input_size = (224, 224)
    
    train_transform = MedicalImageAugmentation.get_medium_augmentation(input_size)
    val_transform = MedicalImageAugmentation.get_test_transform(input_size)
    test_transform = MedicalImageAugmentation.get_test_transform(input_size)
    
    print("✅ 数据增强策略创建完成")
    
    # 3. 创建数据集（示例）
    # train_dataset = Dataset(
    #     data_dir="data/data_split/train",
    #     transform=train_transform,
    #     cache_images=False  # 根据内存情况决定
    # )
    
    # 4. 创建数据加载器
    # train_loader = DataLoader.create_dataloader(
    #     dataset=train_dataset,
    #     batch_size=16,
    #     shuffle=True,
    #     use_weighted_sampler=True  # 处理类别不平衡
    # )
    
    print("✅ 预处理流程演示完成")


if __name__ == "__main__":
    run_preprocessing()