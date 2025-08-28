"""
简化版预测脚本 - 对单张图像进行预测
"""

import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms

from config import Config
from model import create_model

class Predictor:
    """预测器"""
    
    def __init__(self, model_path=None):
        self.model_path = model_path or os.path.join(Config.WEIGHT_DIR, "best_model.pth")
        self.model = None
        self.class_to_idx = None
        self.idx_to_class = None
        self.transform = None
        
        self.load_model()
        self.setup_transform()
    
    def load_model(self):
        """加载模型"""
        if not os.path.exists(self.model_path):
            print(f"模型文件不存在: {self.model_path}")
            print("请先训练模型")
            return
        
        print(f"加载模型: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=Config.DEVICE)
        
        # 创建模型
        self.model = create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 加载类别映射
        self.class_to_idx = checkpoint['class_to_idx']
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        print(f"模型加载成功，支持 {len(self.class_to_idx)} 个类别")
    
    def setup_transform(self):
        """设置图像变换"""
        self.transform = transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict_image(self, image_path, top_k=5):
        """预测单张图像"""
        if self.model is None:
            print("模型未加载")
            return None
        
        if not os.path.exists(image_path):
            print(f"图像文件不存在: {image_path}")
            return None
        
        try:
            # 加载和预处理图像
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(Config.DEVICE)
            
            # 预测
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                # 获取top-k预测
                top_probs, top_indices = torch.topk(probabilities, top_k)
                top_probs = top_probs.cpu().numpy()[0]
                top_indices = top_indices.cpu().numpy()[0]
                
                # 构建结果
                results = []
                for i in range(top_k):
                    class_idx = top_indices[i]
                    class_name = self.idx_to_class[class_idx]
                    probability = top_probs[i]
                    results.append({
                        'class': class_name,
                        'probability': probability,
                        'confidence': f"{probability*100:.2f}%"
                    })
                
                return results
                
        except Exception as e:
            print(f"预测图像时出错: {e}")
            return None
    
    def predict_batch(self, image_dir, output_file=None):
        """批量预测目录中的图像"""
        if self.model is None:
            print("模型未加载")
            return
        
        if not os.path.exists(image_dir):
            print(f"目录不存在: {image_dir}")
            return
        
        # 获取所有图像文件
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend([f for f in os.listdir(image_dir) if f.lower().endswith(ext)])
        
        if not image_files:
            print(f"目录中没有找到图像文件: {image_dir}")
            return
        
        print(f"找到 {len(image_files)} 张图像，开始批量预测...")
        
        results = []
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            prediction = self.predict_image(image_path, top_k=1)
            
            if prediction:
                result = {
                    'image': image_file,
                    'predicted_class': prediction[0]['class'],
                    'confidence': prediction[0]['confidence']
                }
                results.append(result)
                print(f"{image_file}: {prediction[0]['class']} ({prediction[0]['confidence']})")
        
        # 保存结果到文件
        if output_file and results:
            import csv
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['image', 'predicted_class', 'confidence'])
                writer.writeheader()
                writer.writerows(results)
            print(f"预测结果已保存到: {output_file}")
        
        return results

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='支气管分类预测')
    parser.add_argument('--image', type=str, help='单张图像路径')
    parser.add_argument('--dir', type=str, help='图像目录路径（批量预测）')
    parser.add_argument('--model', type=str, help='模型文件路径')
    parser.add_argument('--output', type=str, help='输出文件路径（批量预测时）')
    parser.add_argument('--top_k', type=int, default=5, help='显示前k个预测结果')
    
    args = parser.parse_args()
    
    # 创建预测器
    predictor = Predictor(args.model)
    
    if args.image:
        # 单张图像预测
        print(f"预测图像: {args.image}")
        results = predictor.predict_image(args.image, args.top_k)
        
        if results:
            print("\n预测结果:")
            print("-" * 50)
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['class']}: {result['confidence']}")
        
    elif args.dir:
        # 批量预测
        print(f"批量预测目录: {args.dir}")
        predictor.predict_batch(args.dir, args.output)
        
    else:
        print("请指定 --image 或 --dir 参数")
        print("使用示例:")
        print("  python predict.py --image path/to/image.jpg")
        print("  python predict.py --dir path/to/images/ --output results.csv")

if __name__ == "__main__":
    main()