import shutil
import time

import cv2
import torchvision

from torchvision import models
import torch
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import os

from tqdm import tqdm
from torchmetrics import F1Score, Accuracy, Recall, Precision, Specificity, ConfusionMatrix
import sklearn.metrics
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay

from Arc import ArcMarginProduct, ArcModel
from train import TrainConfig
from data_process import (loadtraindata, loadtestdata, split_data)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    # -------------------------------------------------------------------------------------------
    '''加载权重，并打印加载信息'''
    model_weights_path = [
        # (mymodels.resnet50, r"./weight/resnet50_bestacc0.6097.pth"),
        # (mymodels.resnet50, r"./weight/resnet50_bestacc0.7553.pth"),
        # (mymodels.resnet50, r"./weight/resnet50_bestacc0.8358.pth"),
        # (mymodels.efficientnet_b0, r"./weight/efficientnet_b0_bestacc0.7310.pth"),
        # (mymodels.efficientnet_b0, r"./weight/efficientnet_b0_bestacc0.8394.pth"),
        (models.convnext_tiny, r"I:\zsn\code\classification\weight\convnext_tiny+eca+arc_bestacc0.9366.pth")
        # (mymodels.efficientnet_b3, r"./weight/efficientnet_b3_bestacc0.8558.pth"),
    ]

    for i in range(0, len(model_weights_path)):
        config = TrainConfig(model=model_weights_path[i][0])

        net = config.model(num_classes=config.cls_num)
        weights_path = model_weights_path[i][1]

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

        loaded_ckpt = torch.load(weights_path)
        # print(loaded_ckpt['train_config'])
        # print(loaded_ckpt['ACCURACY_TRAIN'])
        # print(f"best_acc: {loaded_ckpt['best_acc']}")
        weights = loaded_ckpt['best_weight']
        w = {}
        for k, v in weights.items():
            if net.state_dict()[k].shape == v.shape:
                w[k] = v
        print(net.load_state_dict(w, strict=True))

        net.to(config.device)
        testloader = loadtestdata(
            config.testdata_path,
            # config.valdata_path,
            16, config.input_size, config)

        img_path = r'I:\zsn\code\classification\datav4\valid\16_LB5'

        T = torchvision.transforms.Compose([
            torchvision.transforms.Resize(config.input_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # 输入标准化

        # test_f1_macro = F1Score(task='multiclass', num_classes=config.cls_num, average="macro").to(
        #     config.device)  # F1 score
        # test_f1_micro = F1Score(task='multiclass', num_classes=config.cls_num, average="micro").to(
        #     config.device)  # F1 score
        # test_acc = Accuracy(task='multiclass', num_classes=config.cls_num, ).to(config.device)  # Accuracy
        # test_R = Recall(task='multiclass', num_classes=config.cls_num, ).to(config.device)  # Recall
        # test_P = Precision(task='multiclass', num_classes=config.cls_num, ).to(config.device)  # Precision
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

        cls = testloader.dataset.classes

        # 存储预测错误的图片
        err_img = []

        with torch.no_grad():
            for i in os.listdir(img_path):
                img = PIL.Image.open(os.path.join(img_path, i))
                inputs, labels = T(img).unsqueeze(0), torch.tensor(cls.index(os.path.basename(img_path)))
                inputs, labels = inputs.to(config.device), labels.to(config.device)
                if config.use_arc:
                    # outputs = net(inputs, labels)
                    outputs = net(inputs, labels, training=False)
                else:
                    outputs = net(inputs)
                predict = torch.max(outputs, 1)[1].data.squeeze()
                # img_test_num += len(labels)
                # img_test_T += (predict == labels).sum().item()
                all_pred.append(predict)
                all_label.append(labels)
                if predict.item() != labels.item():
                    print("图片: " + img.filename + " 错误预测为: " + cls[predict])
                    err_img.append(img.filename)
                    # 保存预测错误图像
                    img.save(os.path.join(rf"C:\Users\BEgroup\Desktop\1",
                                                         str(cls[predict]))+ "-" +str(len(err_img)) + ".jpg")

        all_preds = torch.stack(all_pred)
        all_labels = torch.stack(all_label)
        acc = (all_preds == all_labels).sum().item() / len(all_pred)
        print("预测准确率: " + str(acc))
        print(len(err_img))
