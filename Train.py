#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file         :Train.py
@description  :
@time         :2020/11/22 23:11:34
@author       :wizz
@version      :1.0
'''


import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# for linux server without GUI
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Model_GRU import GRU
from DataLoad import train_dataset, INPUT_SIZE, PRED_STEP, validation_dataset


# evaluate in RMSE
def rmse(predictions, target):
    return np.sqrt(np.square(predictions - target).mean())


EPOCH = 100
NUM_LAYERS = 2


model = GRU(input_size=INPUT_SIZE, hidden_size=12,
            num_layers=NUM_LAYERS, output_size=PRED_STEP*2)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # weight_decay=1e-3)
loss_list = []
actual_pred = []
ground_truth = []
rmse_list = []

start = datetime.datetime.now()
print('{}   start training...'.format(start))
for epoch in range(EPOCH):

    # train
    actual_pred.clear()
    ground_truth.clear()
    sum_loss = 0.0
    for inputs, labels in train_dataset:
        optimizer.zero_grad()
        model.hidden_cell = torch.zeros(NUM_LAYERS, 1, model.hidden_size)
        y_out = model(inputs)
        # ps:将labels平铺为一维数组！
        labels = torch.flatten(labels)
        loss = loss_function(y_out, labels)
        sum_loss += loss.item()
        loss.backward()
        optimizer.step()
    loss_list.append(sum_loss/len(train_dataset))
    print('{}   epoch:{}    loss:{}'.format(
        datetime.datetime.now(), epoch + 1, loss_list[-1]))

    # test in validation dataset
    for inputs, labels in validation_dataset:
        labels = torch.flatten(labels)
        ground_truth.extend(labels.numpy().tolist())
        with torch.no_grad():
            # 注：这里验证时hidden_cell初始化采用了全零，对预测结果会有影响
            model.hidden_cell = torch.zeros(NUM_LAYERS, 1, model.hidden_size)
            prediction = model(inputs).numpy().tolist()
            actual_pred.extend(prediction)

    actual_pred = np.array(actual_pred).reshape(-1, 1)
    ground_truth = np.array(ground_truth).reshape(-1, 1)

    # calculate rmse
    rmse_val = rmse(actual_pred, ground_truth)
    rmse_list.append(rmse_val)
    actual_pred = actual_pred.tolist()
    ground_truth = ground_truth.tolist()

    if epoch % 5 == 4:
        # 保存模型到文件
        torch.save(model.state_dict(),
                   './model/' + '%d' % (epoch + 1) + '_GRU.pth')
        print('save model successfully !!!')
        # 绘制训练过程损失变化曲线图
        plt.title('Train loss on GRU')
        plt.xlabel('epoch')
        plt.ylabel('train_loss')
        plt.grid(True)
        plt.plot(np.arange(1, len(loss_list) + 1), np.array(loss_list), 'k')
        plt.savefig('./train/train_loss.png', format='png', dpi=200)
        plt.close()
        # 绘制验证集上的rmse
        plt.title('Validation RMSE on GRU')
        plt.xlabel('epoch')
        plt.ylabel('RMSE')
        plt.grid(True)
        plt.plot(np.arange(1, len(rmse_list) + 1), np.array(rmse_list), 'k')
        plt.savefig('./train/validation_rmse.png', format='png', dpi=200)
        plt.close()


end = datetime.datetime.now()
print('Training costs time: {}'.format(end - start))
print('train model completely !!!')
