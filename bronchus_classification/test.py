import datetime
import shutil

import cv2
from timm.models.efficientnet import efficientnet_b0

from torchvision import models
import torch
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import os
# from efficientnet_pytorch import EfficientNet
import model.convnext
from model import vision_transformer, efficientnet_v2
from tqdm import tqdm
from loguru import logger
from torchmetrics import F1Score, Accuracy, Recall, Precision, Specificity, ConfusionMatrix
import sklearn.metrics
from sklearn.metrics import (confusion_matrix, accuracy_score, balanced_accuracy_score,
                             classification_report,precision_score, recall_score, f1_score,)

from sklearn.metrics import ConfusionMatrixDisplay

from config import TrainConfig
from Arc import ArcMarginProduct, ArcModel
from data_process import (loadtraindata, loadtestdata, split_data)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    # -------------------------------------------------------------------------------------------
    '''加载权重，并打印加载信息'''
    model_weights_path = [
        (models.convnext_tiny, r"D:\classification\weight\train_202508161119\convnext_tiny_bestacc0.9789.pth"),
    ]
    for i in range(0, len(model_weights_path)):
        config = TrainConfig(model=model_weights_path[i][0])
        '''设定数据加载器'''
        net = config.model(num_classes=config.cls_num)
        # net = config.model
        if config.use_arc:
            # arc_head = ArcMarginProduct(net.classifier[1].in_features, config.cls_num,
            #                             config.arc_config['s'], config.arc_config['m'],
            #                             config.arc_config['easy_margin'], )
            # net.classifier = torch.nn.Sequential()
            # net = ArcModel(net, arc_head)
            arc_head = ArcMarginProduct(net.classifier[2].in_features, config.cls_num,
                                        config.arc_config['s'], config.arc_config['m'],
                                        config.arc_config['easy_margin'], )
            del net.classifier[2]
            net = ArcModel(net, arc_head)

        weights_path = model_weights_path[i][1]
        loaded_ckpt = torch.load(weights_path)

        '''创建日志保存路径'''
        cur_time = datetime.datetime.now().strftime('test_%Y%m%d%H%M')  # 记录测试开始时间
        output_dir = os.path.join(config.save_path, cur_time)  # 创建测试文件保存文件夹
        os.makedirs(output_dir, exist_ok=True)
        # 生成日志文件
        log_path = os.path.join(output_dir, os.path.basename(weights_path).replace('.pth', '.log'))
        logger.add(log_path)
        logger.info(f'weights_path:{weights_path}')

        print(loaded_ckpt['train_config'])
        print(loaded_ckpt['ACCURACY_TRAIN'])
        logger.info(f"best_acc: {loaded_ckpt['best_acc']}")

        weights = loaded_ckpt['best_weight']
        w = {}
        for k, v in weights.items():
            if net.state_dict()[k].shape == v.shape:
                w[k] = v
        print(net.load_state_dict(w, strict=True))

        # net = torch.load(r'F:\liwenlong\Awesome-Backbones\tools\cspdarknet50.pth')

        net.to(config.device)

        testloader = loadtestdata(
            config.testdata_path,
            # config.valdata_path,
            16, config.input_size, config)

        # test_f1_macro = F1Score(task='multiclass', num_classes=config.cls_num, average="macro").to(
        #     config.device)  # F1 score
        # test_f1_micro = F1Score(task='multiclass', num_classes=config.cls_num, average="micro").to(
        #     config.device)  # F1 score
        # test_acc = Accuracy(task='multiclass', num_classes=config.cls_num, ).to(config.device)  # Accuracy
        test_R = Recall(task='multiclass', num_classes=config.cls_num, ).to(config.device)  # Recall
        test_P = Precision(task='multiclass', num_classes=config.cls_num, ).to(config.device)  # Precision
        # test_spc = Specificity(task='multiclass', num_classes=config.cls_num, ).to(config.device)   # Specificity
        # test_conf_mat = ConfusionMatrix(task='multiclass', num_classes=config.cls_num, ).to(
        #     config.device)  # Confusion Matrix
        img_train_T = 0.0
        img_test_T = 0.0
        sum_train_loss = []
        sum_test_loss = []
        img_train_num = 0
        img_test_num = 0

        all_pred = []
        all_label = []
        net.eval()
        with torch.no_grad():
            for i, data in tqdm(enumerate(testloader, 0)):
                inputs, labels = data
                inputs, labels = inputs.to(config.device), labels.to(config.device)
                if config.use_arc:
                    # outputs = net(inputs, labels)
                    outputs = net(inputs, labels, training=False)
                else:
                    outputs = net(inputs)
                predict = torch.max(outputs, 1)[1].data.squeeze()

                img_test_num += len(labels)
                img_test_T += (predict == labels).sum().item()
                all_pred.append(predict)
                all_label.append(labels)

        cls = testloader.dataset.classes

        '''id sort'''
        def cmp(x):
            return int(x.split('_')[0])
        cls_sort = sorted(cls, key=cmp)

        index_sort = {}
        for i in range(len(cls_sort)):
            j = cls.index(cls_sort[i])
            index_sort[j] = i

        all_pred = torch.cat(all_pred).cpu().numpy()
        all_label = torch.cat(all_label).cpu().numpy()
        all_pred_ = np.array([index_sort[i] for i in all_pred])
        all_label_ = np.array([index_sort[i] for i in all_label])
        epoch_test_acc = accuracy_score(all_label_, all_pred_)
        epoch_test_acc_bla = balanced_accuracy_score(all_label_, all_pred_)

        cm = confusion_matrix(all_label_, all_pred_)
        # epoch_test_f1_macro = test_f1_macro(all_pred, all_label) # F1 score
        # epoch_test_f1_micro = test_f1_micro(all_pred, all_label)
        # epoch_test_acc = test_acc(all_pred, all_label)
        # epoch_test_R = test_P(all_pred, all_label)
        # epoch_test_P = test_R(all_pred, all_label)
        # cm = test_conf_mat(all_pred, all_label)
        # 召回率 精确率
        # epoch_test_R = test_P(all_pred, all_label)
        # epoch_test_P = test_R(all_pred, all_label)

        epoch_test_R = precision_score(all_label, all_pred, average='macro')
        epoch_test_P = recall_score(all_label, all_pred, average='macro')
        epoch_test_F = f1_score(all_label, all_pred, average='macro')
        print('acc', epoch_test_acc)
        print('test_P',epoch_test_P)
        print('test_R',epoch_test_R)
        print('test_f1_score',epoch_test_F)

        '''保存预测错误的图像'''
        for j in range(len(all_pred_)):
            if all_pred[j] != all_label[j]:
                err_img_name = os.path.basename(testloader.dataset.imgs[j][0]).replace('.jpg',
                                                                                       f'_{all_label[j]}to{all_pred[j]}.jpg')
                dest_path = os.path.join(output_dir, err_img_name)
                shutil.copy(testloader.dataset.imgs[j][0], dest_path)
                logger.info("预测错误的图像:" + testloader.dataset.imgs[j][0] +
                            "    真实标签:" + str(all_label[j]) + "   预测标签:" + str(all_pred[j]))

        '''画混淆矩阵'''
        fontsize = 10
        fig, ax = plt.subplots(figsize=(fontsize, fontsize))
        im = ax.imshow(cm)
        ax.set_xticks(np.arange(len(cls_sort)))
        ax.set_yticks(np.arange(len(cm)))
        # ax.set_xticks(np.arange(0, len(cls)*2, 2))
        # ax.set_yticks(np.arange(0, len(a)*2, 2))
        ax.set_xticklabels(cls_sort, fontsize=fontsize)
        ax.set_yticklabels(cls_sort, fontsize=fontsize)
        ax.set_xlabel('predict', fontsize=fontsize)
        ax.set_ylabel('true', fontsize=fontsize)
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right",
                 rotation_mode="anchor")
        report = classification_report(all_label_, all_pred_, output_dict=True)
        cls_p = []
        cls_r = []
        cls_f1 = []
        cls_n = []
        for i in range(config.cls_num):
            cls_p.append(report[str(i)]['precision'])
            cls_r.append(report[str(i)]['recall'])
            cls_f1.append(report[str(i)]['f1-score'])
            cls_n.append(report[str(i)]['support'])
        print('cls_p\n', str(np.array(cls_p).round(4).tolist())[1:-1].replace(',', ''))
        print('cls_r\n', str(np.array(cls_r).round(4).tolist())[1:-1].replace(',', ''))
        print('cls_f1\n', str(np.array(cls_f1).round(4).tolist())[1:-1].replace(',', ''))
        print('cls_n\n', str(np.array(cls_n).round(4).tolist())[1:-1].replace(',', ''))

        for i in range(len(cm)):
            for j in range(len(cls_sort)):
                text = ax.text(j, i, cm[i, j],
                               ha="center", va="center", color="w", fontsize=fontsize)
        ax.set_title(f"{config.model.__name__} Bronchoscopy Classification ConfusionMatrix", fontsize=fontsize)
        fig.tight_layout()

        # 保存混淆矩阵
        fig_path = os.path.join(output_dir,
                                os.path.basename(weights_path).replace('.pth', f'_test{epoch_test_acc:.4f}.jpg'))
        plt.savefig(fig_path)
        plt.show()

        # 计算各类的准确率acc
        accuracy_per_class = []
        for i in range(len(cm)):
            TP = cm[i, i]
            FP = np.sum(cm[:, i]) - TP  # 该列的总和减去对角线上的TP
            total = np.sum(cm[i, :])  # 该行的总和，即该类别的所有预测值
            accuracy = TP / total if total != 0 else 0  # 如果该类别的总预测数为0，避免除零错误
            accuracy_per_class.append(accuracy)

        # 输出各类别的准确率
        for idx, accuracy in enumerate(accuracy_per_class):
            print(f"Class {idx} accuracy: {accuracy:.4f}")
