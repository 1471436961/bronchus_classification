"""
简化版数据处理模块
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import Config

class BronchusDataset(Dataset):
    """支气管数据集"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        
        self._load_samples()
    
    def _load_samples(self):
        """加载样本"""
        if not os.path.exists(self.data_dir):
            print(f"警告: 数据目录不存在 {self.data_dir}")
            return
            
        classes = sorted([d for d in os.listdir(self.data_dir) 
                         if os.path.isdir(os.path.join(self.data_dir, d))])
        
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        
        for class_name in classes:
            class_dir = os.path.join(self.data_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
        
        print(f"加载数据: {len(self.samples)} 个样本, {len(classes)} 个类别")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"加载图像失败: {img_path}, 错误: {e}")
            # 返回一个默认图像
            image = Image.new('RGB', (Config.IMAGE_SIZE, Config.IMAGE_SIZE), (0, 0, 0))
            if self.transform:
                image = self.transform(image)
            return image, label

def get_transforms(is_training=True):
    """获取数据变换"""
    if is_training and Config.USE_AUGMENTATION:
        transform = transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE + 32, Config.IMAGE_SIZE + 32)),
            transforms.RandomCrop(Config.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform

def create_dataloaders():
    """创建数据加载器"""
    # 训练数据
    train_transform = get_transforms(is_training=True)
    train_dataset = BronchusDataset(Config.TRAIN_DIR, train_transform)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE.type == 'cuda' else False
    )
    
    # 验证数据
    val_transform = get_transforms(is_training=False)
    val_dataset = BronchusDataset(Config.VAL_DIR, val_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE.type == 'cuda' else False
    )
    
    # 测试数据
    test_dataset = BronchusDataset(Config.TEST_DIR, val_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE.type == 'cuda' else False
    )
    
    return train_loader, val_loader, test_loader, train_dataset.class_to_idx

if __name__ == "__main__":
    # 测试数据加载
    train_loader, val_loader, test_loader, class_to_idx = create_dataloaders()
    print(f"类别映射: {class_to_idx}")
    print(f"训练集: {len(train_loader)} 批次")
    print(f"验证集: {len(val_loader)} 批次")
    print(f"测试集: {len(test_loader)} 批次")