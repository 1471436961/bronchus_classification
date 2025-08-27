import torchvision
from torchvision import models
import os


class TrainConfig():
    '''
    设定训练时模型参数
    '''
    def __init__(self, model=None):
        self.device = 'cuda'
        '''输入数据'''

        self.cls_num = 33                                   # 定义分类类别数量
        self.traindata_path = rf"D:\classification\data\X1_split\train"      # 训练集路径
        self.valdata_path = rf"D:\classification\data\X1_split\val"        # 验证集路径
        self.testdata_path = rf"D:\classification\data\X1_split\test"        # 测试集路径
        self.save_path = './weight'                         # 输出父级路径
        os.makedirs(self.save_path, exist_ok=True)

        '''模型选择'''
        self.model = models.efficientnet_b0 if model == None else model  # model为构造时传入的参数

        # 是否使用arcface损失
        self.use_arc =False
        # self.use_arc = True
        self.arc_config = {'s': 15.0, 'm': 0.30, 'easy_margin': False}



        self.is_pretrain = True  # 是否使用预训练权重

        self.input_size = (224, 224)  # 调整输入图像尺寸
        if self.model == models.efficientnet_b3:
            self.input_size = (300, 300)  # 调整输入图像尺寸
        elif self.model == models.inception_v3:
            self.input_size = (299, 299)

        self.is_data_aug = True  # 是否做数据增强

        # Wu数据增强
        # aug0
        # self.aug = torchvision.transforms.Compose([
        #     torchvision.transforms.Resize(self.input_size),  # 图像尺寸固定，默认大小为(224, 224)
        #     # 随机仿射变换
        #     torchvision.transforms.ToTensor(),
        #     torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]  # 输入标准化（RGB三通道均值和方差）
        # )
        # ([0.5679336, 0.34756106, 0.26917857], [0.22522911, 0.18319885, 0.1429226])

        # 自定义数据增强
        # aug1
        # self.aug = torchvision.transforms.Compose([
        #     torchvision.transforms.Resize(self.input_size),  # 图像尺寸固定，默认大小为(224, 224)
        #     # 随机仿射变换
        #     torchvision.transforms.RandomAffine(degrees=(-45, 45), translate=(0.1, 0.1), scale=(0.9, 1.1),
        #                                         fill=(0, 0, 0)),
        #     # 颜色抖动
        #     torchvision.transforms.ColorJitter(brightness=(0.8, 1.3), contrast=(0.85, 1.5)),
        #     torchvision.transforms.ToTensor(),
        #     torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]  # 输入标准化（RGB三通道均值和方差）
        # )

        # 自定义数据增强加入高斯噪声
        self.aug = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.input_size),  # 图像尺寸固定，默认大小为(224, 224)
            # 随机仿射变换
            torchvision.transforms.RandomAffine(degrees=(-30, 30), translate=(0.1, 0.1), scale=(0.9, 1.2),
                                                fill=(0, 0, 0)),
            # 颜色抖动
            torchvision.transforms.ColorJitter(brightness=(0.8, 1.3), contrast=(0.85, 1.4)),
            torchvision.transforms.RandomApply([torchvision.transforms.GaussianBlur((5, 5), sigma=(0.1, 5))], p=0.3),
            torchvision.transforms.Pad(5, (0, 0, 0)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 输入标准化（RGB三通道均值和方差）
        ])

        # TrivialAugmentWide()
        # self.aug = torchvision.transforms.Compose([
        #     torchvision.transforms.TrivialAugmentWide(),
        #     torchvision.transforms.Resize(self.input_size),  # 图像尺寸固定，默认大小为(224, 224)
        #     torchvision.transforms.ToTensor(),
        #     torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 输入标准化（RGB三通道均值和方差）
        # ])

        '''ckpt'''
        # self.ckpt = None
        self.ckpt = './weight/train_202508132013/convnext_tiny_bestacc0.9416.pth'


        '''实验配置设置'''
        self.start_epoch = 0
        self.epochs = 50  # 训练轮次
        self.optimizer = 'adam'
        self.init_lr = 1e-4
        self.milestones = [15, 30, 45]
        self.batch_size = 8  # 调整batch_size
