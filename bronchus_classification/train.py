# ----------------------------------------------------------------------------------------------------------------------
# 加载必要的第三方包
# ----------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
from torchvision import models
from torchsummary import summary
import torch.optim as optim
import torch.nn as nn
import torchvision
import torch, copy
from torch.utils.data import DataLoader
from loguru import logger
import datetime
from torchmetrics import F1Score, Accuracy, Recall, Precision, Specificity, ConfusionMatrix
import os
import timm

import loss.focal_loss
import model.convnext, model.vision_transformer,model.swin_transformer
from Arc import ArcMarginProduct, ArcModel
from data_process import (loadtraindata, loadtestdata, split_data)
from config import TrainConfig

# from .Blocks.focal_loss import FocalLoss
import numpy as np
# from efficientnet_pytorch import EfficientNet
# from dataset import MyDataset


# 不添加该语句程序可能会出现OMP错误
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# from Arc import ArcMarginProduct, ArcModel

# 加载预训练权重
import pre_model

# 训练主函数
def train(config):
    # 记录训练设置参数
    logger.info('*' * 50)
    logger.info(config.__dict__)
    logger.info('*' * 50)

    best_metric = []
    best_acc = -1
    best_epoch = -1
    best_weight = None

    EPOCH = []
    ACCURACY_TRAIN = []
    ACCURACY_TEST = []
    LOSS_TRAIN = []
    LOSS_TEST = []

    test_f1_macro = F1Score(task='multiclass', num_classes=config.cls_num, average="macro").to(
        config.device)  # F1 score
    test_f1_micro = F1Score(task='multiclass', num_classes=config.cls_num, average="micro").to(
        config.device)  # F1 score
    test_acc = Accuracy(task='multiclass', num_classes=config.cls_num, ).to(config.device)  # Accuracy
    test_R = Recall(task='multiclass', num_classes=config.cls_num, ).to(config.device)  # Recall
    test_P = Precision(task='multiclass', num_classes=config.cls_num, ).to(config.device)  # Precision
    # test_spc = Specificity(task='multiclass', num_classes=config.cls_num, ).to(config.device)   # Specificity
    test_conf_mat = ConfusionMatrix(task='multiclass', num_classes=config.cls_num, ).to(
        config.device)  # Confusion Matrix

    # ------------------------------------------------------------------------------------------------------------------
    # 迭代epochs
    # sum_test_acc = 0.0 n_test = 0 train_acc = 0 sum_train_loss sum_test_loss --- 参数初始化
    # ------------------------------------------------------------------------------------------------------------------

    for epoch in range(config.start_epoch, config.epochs):
        img_train_T = 0.0
        img_test_T = 0.0
        sum_train_loss = []
        sum_test_loss = []
        img_train_num = 0
        img_test_num = 0

        all_pred = []
        all_label = []
        # --------------------------------------------------------------------------------------------------------------
        # 加载训练数据集进行训练
        # .to(device) --- 将数据读取到cpu或者gpu
        # optimizer.zero_grad() --- 每次喂入数据前，都需要将梯度清零
        # net(inputs) --- 将数据喂入神经网络 --- 得到结果
        # criterion(outputs, labels) --- 得到loss
        # loss.backward() --- 反向传播
        # optimizer.step() --- 调用optimizer.step()会使优化器迭代它应该更新的所有参数(张量),并使用它们内部存储的grad来更新它们的值
        # torch.max(outputs, 1)[1].data.squeeze() --- 得到预测的标签
        # train_acc = (predict == labels).sum().item() / labels.size(0) --- 得到最后一批训练数据的准确率
        # --------------------------------------------------------------------------------------------------------------

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            optimizer.zero_grad()
            if config.use_arc:
                outputs = net(inputs, labels)
            else:
                outputs = net(inputs)
            if config.model.__name__ == 'GoogLeNet':
                losses = []
                for output in outputs:  # googlenet含有两个辅助损失
                    losses.append(criterion(output, labels))
                loss = sum(losses) / len(outputs)
                # loss = losses[0]
                outputs = outputs[0]
            elif config.model.__name__ == 'inception_v3':
                outputs = outputs[0]
                loss = criterion(outputs, labels)
            else:
                # loss = criterion(outputs.softmax(1), labels)
                # 增加损失项
                loss = criterion(outputs, labels)
                # todo: 增加损失想
                # loss = 0.5 * criterion(outputs, labels) + 0.5* focal_loss(outputs, labels)

            loss.backward()
            optimizer.step()

            predict = torch.max(outputs, 1)[1].data.squeeze()
            train_acc = (predict == labels).sum().item() / len(labels)
            # 每50个batch打印训练信息
            if i % 50 == 0:
                logger.info('[%d, %5d] Train Loss: %.4f Train Accuracy: %.4f ' % (
                    epoch + 1, (i + 1) * config.batch_size, loss.item(), train_acc))
            img_train_num += len(labels)
            img_train_T += (predict == labels).sum().item()
            sum_train_loss.append(loss.item())

        scheduler.step()

        # --------------------------------------------------------------------------------------------------------------
        # 测试过程，
        # --------------------------------------------------------------------------------------------------------------

        net.eval()
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(config.device), labels.to(config.device)
                if config.use_arc:
                    outputs = net(inputs, labels, training=False)
                else:
                    outputs = net(inputs)
                # todo 联合损失
                loss = criterion(outputs, labels)
                # loss = 0.5 * criterion(outputs, labels) + 0.5* focal_loss(outputs, labels)
                predict = torch.max(outputs, 1)[1].data.squeeze()

                img_test_num += len(labels)
                img_test_T += (predict == labels).sum().item()
                all_pred.append(predict)
                all_label.append(labels)

                sum_test_loss.append(loss.item())
        net.train()

        # --------------------------------------------------------------------------------------------------------------
        # 输出训练/测试的信息
        # --------------------------------------------------------------------------------------------------------------
        all_pred = torch.cat(all_pred)
        all_label = torch.cat(all_label)
        epoch_test_f1_macro = test_f1_macro(all_pred, all_label)  # F1 score
        epoch_test_f1_micro = test_f1_micro(all_pred, all_label)
        # 计算测试损失
        epoch_test_acc = test_acc(all_pred, all_label)
        epoch_test_R = test_P(all_pred, all_label)
        epoch_test_P = test_R(all_pred, all_label)

        epoch_train_acc = img_train_T / img_train_num
        epoch_train_loss = sum(sum_train_loss) / len(sum_train_loss)
        epoch_test_loss = sum(sum_test_loss) / len(sum_test_loss)
        # logger.info('epoch: [%d] train_acc: %.4f test_acc: %.4f train_loss: %.4f test_loss: %.4f' %
        #             (epoch + 1, epoch_train_acc, epoch_test_acc, epoch_train_loss, epoch_test_loss))
        logger.info('epoch: [%d] train_acc: %.4f test_acc: %.4f train_loss: %.4f test_loss: %.4f learning_rate: %f' %
                    (epoch + 1, epoch_train_acc, epoch_test_acc, epoch_train_loss, epoch_test_loss,
                     optimizer.state_dict()['param_groups'][0]['lr']))

        # --------------------------------------------------------------------------------------------------------------
        # 将信息添加到列表
        # --------------------------------------------------------------------------------------------------------------

        EPOCH.append(epoch + 1)
        ACCURACY_TRAIN.append(epoch_train_acc)
        ACCURACY_TEST.append(epoch_test_acc.item())
        LOSS_TRAIN.append(epoch_train_loss)
        LOSS_TEST.append(epoch_test_loss)

        if epoch_test_acc > best_acc:
            best_epoch = epoch
            best_weight = copy.deepcopy(net.state_dict())
            best_acc = epoch_test_acc
            best_metric = [epoch_test_f1_macro,
                           epoch_test_f1_micro,
                           epoch_test_acc,
                           epoch_test_R,
                           epoch_test_P]
            best_label_pred = [all_label, all_pred]

        '''训练完成后的最佳结果及日志,可用于绘图'''
        ckpt_state = {
            'train_config': config.__dict__,
            'best_epoch': best_epoch,
            "best_weight": best_weight,
            'best_acc': best_acc,
            'best_metric': best_metric,
            'best_label_pred': best_label_pred,
            'ACCURACY_TRAIN': ACCURACY_TRAIN,
            'ACCURACY_TEST': ACCURACY_TEST,
            'LOSS_TRAIN': LOSS_TRAIN,
            'LOSS_TEST': LOSS_TEST,
        }
        logger.info('-' * 50)
        # logger.info(config.__dict__)
        logger.info(best_metric)
        logger.info(f'best_acc: {best_acc}')
        logger.info('-' * 50)

        '''设定模型训练结果及日志的保存'''
        config.model_name = f'{config.model.__name__}'
        save_path = os.path.join(output_dir, f'{config.model_name}_bestacc{best_acc:.4f}.pth')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(ckpt_state, save_path)

    # --------------------------------------------------------------------------------------------------------------
    # 信息可视化
    # --------------------------------------------------------------------------------------------------------------
    plt_acc_path = os.path.join(output_dir, f'{config.model_name}{cur_time}_bestacc{best_acc:.4f}_acc.png')
    plt.plot(EPOCH, ACCURACY_TRAIN)
    plt.plot(EPOCH, ACCURACY_TEST)
    plt.title('Accuracy')
    plt.legend(labels=["train", "test"], loc="lower right", fontsize=13)
    plt.savefig(plt_acc_path)
    plt.show()

    # 绘制损失函数
    plt_acc_path = os.path.join(output_dir, f'{config.model_name}{cur_time}_bestacc{best_acc:.4f}_loss.png')
    plt.plot(EPOCH, LOSS_TRAIN)
    plt.plot(EPOCH, LOSS_TEST)
    plt.title('Loss')
    plt.legend(labels=["train", "test"], loc="lower right", fontsize=13)
    plt.savefig(plt_acc_path)
    plt.show()


if __name__ == '__main__':

    '''加载训练配置'''
    config = TrainConfig(models.convnext_tiny)
    config.model.__name__ = 'convnext_tiny'
    # config = TrainConfig(model.swin_transformer.swin_tiny_patch4_window7_224(num_classes = 31))
    cur_time = datetime.datetime.now().strftime('train_%Y%m%d%H%M')  # 记录训练开始时间
    output_dir = os.path.join(config.save_path, cur_time)  # 创建训练文件保存文件夹
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f'{config.model.__name__}{cur_time}.log')  # 训练日志文件保存路径
    logger.add(log_path)
    logger.info(f'log path: {log_path}')

    '''设定数据加载器'''
    trainloader = loadtraindata(config.traindata_path, config.batch_size, config.input_size, config.is_data_aug, config)
    testloader = loadtestdata(config.valdata_path, config.batch_size, config.input_size, config)


    net = config.model(num_classes=config.cls_num)
    # net = config.model

    '''加载预训练权重，并打印加载信息'''
    if config.is_pretrain:
        if config.model.__name__ == 'efficientnetv2_s':
            pretrain_net = config.model()
            pretrain_net.load_state_dict(torch.load(rf'./pre_model/pre_efficientnetv2-s.pth'))
        elif config.model.__name__ == 'convnext_small+simma':
            pretrain_net = config.model()
            # 加载原始与训练参数
            pretrained_state_dict = torch.load(rf'I:\zsn\code\classification\pre_model\convnext_small.pth')
            # 纠正后的预训练权重
            modified_state_dict = {}
            for old_key, value in pretrained_state_dict.items():
                # 依次替换键名
                if 'block.2' in old_key:
                    new_key = old_key.replace('block.2', 'block.3')
                elif 'block.3' in old_key:
                    new_key = old_key.replace('block.3', 'block.4')
                elif 'block.5' in old_key:
                    new_key = old_key.replace('block.5', 'block.6')
                else:
                    new_key = old_key
                modified_state_dict[new_key] = value
            # 加载本地预训练权重
        elif config.model.__name__ == 'efficientnet_b0':
            pretrain_net = config.model()
            pretrain_net.load_state_dict(torch.load(rf'./pre_model/efficientnet-pretrain-b0.pth'))
        else:
            pretrain_net = config.model(pretrained=True)

        # 普通加载
        w = {}
        for k, v in pretrain_net.state_dict().items():
            if net.state_dict()[k].shape == v.shape:
                w[k] = v
        logger.info(net.load_state_dict(w, strict=False))
        logger.info('load pretrain weight completed!')

        #加载更改模型
        # w = {}
        # for k, v in modified_state_dict.items():
        #     if net.state_dict()[k].shape == v.shape:
        #         w[k] = v
        #
        # logger.info(net.load_state_dict(w, strict=False))
        # logger.info('load pretrain weight completed!')

    # '''是否要加上arc损失头'''
    if config.use_arc:
        # 为convnext添加archead
        arc_head = ArcMarginProduct(net.classifier[2].in_features, config.cls_num,
                                    config.arc_config['s'], config.arc_config['m'], config.arc_config['easy_margin'], )

        # arc_head = ArcMarginProduct(net.classifier[1].in_features, config.cls_num,
        # config.arc_config['s'], config.arc_config['m'], config.arc_config['easy_margin'], )
        # net.classifier = nn.Sequential()
        del net.classifier[2]
        net = ArcModel(net, arc_head)
        logger.info('use_arc: Yes!')

    '''加载ckpt'''
    if config.ckpt is not None:
        ckpt = torch.load(config.ckpt)
        net.load_state_dict(ckpt['best_weight'], strict=True)
        # config.milestones = [max(0, config.milestones[0]-ckpt['best_epoch'])]
        # config.start_epoch = ckpt['best_epoch']
        logger.info('load ckpt weight completed!')

    '''设定优化器'''
    if config.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=config.init_lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1, milestones=config.milestones)  # 学习率
    elif config.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=config.init_lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1, milestones=config.milestones)  # 学习率

    '''设定损失函数'''
    # 交叉熵损失
    criterion = nn.CrossEntropyLoss()

    # 加权交叉熵损失
    # criterion = nn.CrossEntropyLoss(
    #     weight=torch.from_numpy(np.array([1.2524, 0.8287, 0.7157, 0.6782, 0.6888, 0.6693, 0.8350, 0.8124, 0.7372,
    #                                       0.8522, 0.8276, 0.7013, 0.8225, 1.3720, 1.0666, 1.9979, 2.0473, 0.9249,
    #                                       1.0876, 1.5167, 1.1980, 1.6288, 0.8997, 0.9914, 1.5272, 1.3691, 0.8633,
    #                                       1.0581, 1.3018, 1.8894, 1.4314])).float().to(config.device))


    logger.info(f'loss: {criterion}')
    net.to(config.device)
    train(config)
