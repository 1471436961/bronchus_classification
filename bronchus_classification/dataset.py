import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import os
import PIL.Image as Image

from data_process import loadtraindata


class MyDataset(Dataset):

    def __init__(self, data_dir, config, is_data_aug=None):
        """
        Args:
            data_dir(string): 数据集根目录路径
            config: 训练配置的实例
            is_data_aug: 是否进行数据增强,是则使用config.aug, 否则使用transform_default

        Attributes:
            label_name: (class_name, class_index)字典
            data_info (list): (样本路径, 类别索引)

        """

        self.label_name = {'1_glottis': 1, '2_trachea': 2, '3_carina': 3, '4_LMB': 4, '5_RMB': 5, '6_LUL': 6,
                           '7_LLL': 7, '8_RUL': 8, '9_RIB': 9,
                           '10_LPB': 10, '11_LLB': 11, '12_RML': 12, '13_RLL': 13, '14_LB1+2': 14, '15_LB3': 15,
                           '16_LB4': 16, '17_LB5': 17, '18_LB6': 18, '19_LB8': 19,
                           '20_LB9': 20, '21_LB10': 21, '22_RB1': 22, '23_RB2': 23, '24_RB3': 24, '25_RB4': 25,
                           '26_RB5': 26, '27_RB6': 27, '28_RB7': 28, '29_RB8': 29, '30_RB9': 30, '31_RB10': 0}

        self.data_info = self.get_img_info(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本

        # 默认的数据增强方法
        self.transform_default = torchvision.transforms.Compose([
            torchvision.transforms.Resize(config.input_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

        self.transform = config.aug if is_data_aug else self.transform_default

    # 根据索引，迭代的读取路径和标签
    def __getitem__(self, index):

        path_img, label = self.data_info[index]  # 索引读取图像路径和标签
        img = Image.open(path_img).convert('RGB')  # 读取图像，返回Image 类型 0~255

        if self.transform is not None:
            img = self.transform(img)  # 在这里做transform，把图像转为tensor等等

        return img, label

    def __len__(self):
        return len(self.data_info)

    # 生成data_info (list): (样本路径, 类别索引)
    def get_img_info(self, data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):  # root:所有文件夹路径 dirs:root下的文件夹路径 files:各文件夹下文件的数组
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))

                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    # 获取该图片的标签
                    label = self.label_name[sub_dir]
                    data_info.append((path_img, int(label)))

        return data_info  # 返回的也就是图像路径和标签

    # trainloader = loadtraindata(config.traindata_path, config.batch_size, config.input_size, config.is_data_aug, config)
    # testloader = loadtestdata(config.valdata_path, config.batch_size, config.input_size, config)

