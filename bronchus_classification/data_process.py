import os, random, shutil
import torchvision
import torch
import PIL.Image
import cv2
from torch.utils.data import DataLoader

'''
加载训练集和测试集所需要的函数
'''


def split_data(org_path, split_path, test_p=0.2):
    '''
    将原始数据划分为训练集和测试集
    org_path:      原始数据集路径
    split_path:    划分后的数据集路径
    test_p:        测试集的比例
    '''

    print(f'{20 * "*"} 开始划分数据集 {20 * "*"}')
    random.seed(0)  # 随机种子固定
    cls_names = os.listdir(org_path)

    '''创建划分后的数据集文件夹'''
    print(f'原路径：{org_path}')
    print(f'划分后路径：{split_path}')
    os.makedirs(split_path, exist_ok=True)  # 新建文件夹

    assert len(cls_names) == 2  # 确认是31类
    for cls_path in cls_names:
        pre_cls_img_all = os.listdir(os.path.join(org_path, cls_path))
        random.shuffle(pre_cls_img_all)
        pre_cls_img_all_num = len(pre_cls_img_all)
        test_num, train_num = int(pre_cls_img_all_num * test_p), int(pre_cls_img_all_num * (1 - test_p))
        train_imgs = pre_cls_img_all[:train_num]
        test_imgs = pre_cls_img_all[-test_num:]
        print(f'{cls_path}  train_num:{train_num} test_num:{test_num}')

        '''复制训练集图像到train文件夹'''
        split_path_train_cls = os.path.join(split_path, f'train/{cls_path}')
        os.makedirs(split_path_train_cls, exist_ok=True)  # 新建文件夹
        for i in train_imgs:
            shutil.copy(os.path.join(org_path, cls_path, i), os.path.join(split_path_train_cls, i))  # 原图像复制到新位置

        '''复制测试集图像到test文件夹'''
        split_path_test_cls = os.path.join(split_path, f'test/{cls_path}')
        os.makedirs(split_path_test_cls, exist_ok=True)  # 新建文件夹
        for i in test_imgs:
            shutil.copy(os.path.join(org_path, cls_path, i), os.path.join(split_path_test_cls, i))  # 原图像复制到新位置
    print(f'{20 * "*"} 数据集划分完成 {20 * "*"}')


def loadtraindata(path, batch_size, input_size, is_data_aug, config):
    # path = "./data/dst/train/"
    # 图像预处理
    # -------------------------------------------------------------------------------------------
    # 数据的加载，torchvision.datasets.ImageFolder是对数据集的处理，
    # 第一个参数数加载数据集的路径，第二个参数是数据集的预处理（将尺寸变为input_size=224x224，转为张量）
    # torch.utils.data.DataLoader --- 载入数据集，上面对数据集的处理，相当于初始化，这里才是对数据集的使用
    # torch.utils.data.DataLoader --- 相当于一个传送带，将数据运送到神经网路
    # 第一个参数是，数据的选择，训练的时候选择训练数据集，测试的时候选择测试数据集，第二个参数是一次传送的数据集个数
    # -------------------------------------------------------------------------------------------
    data_aug_transform = config.aug
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(input_size),  # 图像尺寸固定，默认大小为(224, 224)
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]  # 输入标准化（RGB三通道均值和方差）
    )
    T = data_aug_transform if is_data_aug else transform

    # if 'cls' in config.model.__name__:
    #     T = torchvision.transforms.Compose(T.transforms[:-1])
    # data_aug_transform =

    def BGR_loader(path: str):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        img = cv2.imread(path, 1)
        # with open(path, "rb") as f:
        #     img = PIL.Image.open(f)
        return PIL.Image.fromarray(img)

    trainset = torchvision.datasets.ImageFolder(path,
                                                transform=T,
                                                # loader=BGR_loader,
                                                )
    print(f'trainset classes name and label: {trainset.class_to_idx}')
    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size, num_workers=8, )

    return trainloader 


def loadtestdata(path, batch_size, input_size, config):
    # path = "./data/dst/test/"
    if config.model.__name__ == 'efficientnetv2_s':
        input_size = (384, 384)
    T = torchvision.transforms.Compose([
        torchvision.transforms.Resize(input_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # 输入标准化
    # if 'cls' in config.model.__name__:
    #     T = torchvision.transforms.Compose(T.transforms[:-1])
    testset = torchvision.datasets.ImageFolder(path,
                                               transform=T
                                               )

    print(f'testset classes name and label: {testset.class_to_idx}')
    testloader = DataLoader(testset, shuffle=False, batch_size=batch_size, num_workers=4, )
    return testloader


if __name__ == '__main__':
    split_data(
        r'E:\BranchoData\呼吸AI 吴博士 材料\质控\vivo',
        r'E:\BranchoData\呼吸AI 吴博士 材料\质控\vivo_data',
        test_p=0.2
    )

'''
******************** 开始划分数据集 ********************
原路径：F:\datasets\bronchoscopy\train
划分后路径：F:\datasets\bronchoscopy\train_split
10_LPB      train_num:829 test_num:92
11_LLB      train_num:837 test_num:93
12_RML      train_num:957 test_num:106
13_RLL      train_num:610 test_num:67
14_LB1+2    train_num:538 test_num:59
15_LB3      train_num:733 test_num:81
16_LB4      train_num:396 test_num:44
17_LB5      train_num:337 test_num:37
18_LB6      train_num:763 test_num:84
19_LB8      train_num:623 test_num:69
1_glottis   train_num:221 test_num:24
20_LB9      train_num:494 test_num:54
21_LB10     train_num:624 test_num:69
22_RB1      train_num:479 test_num:53
23_RB2      train_num:715 test_num:79
24_RB3      train_num:730 test_num:81
25_RB4      train_num:477 test_num:53
26_RB5      train_num:512 test_num:56
27_RB6      train_num:748 test_num:83
28_RB7      train_num:691 test_num:76
29_RB8      train_num:543 test_num:60
2_trachea   train_num:548 test_num:60
30_RB9      train_num:394 test_num:43
31_RB10     train_num:534 test_num:59
3_carina    train_num:1167 test_num:129
4_LMB       train_num:1026 test_num:114
5_RMB       train_num:802 test_num:89
6_LUL       train_num:867 test_num:96
7_LLL       train_num:582 test_num:64
8_RUL       train_num:829 test_num:92
9_RIB       train_num:853 test_num:94


vitro  train_num:1440 test_num:360 
vivo  train_num:1680 test_num:420
******************** 数据集划分完成 ********************
'''
