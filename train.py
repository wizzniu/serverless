#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file         :train.py
@description  :
@time         :2020/11/28 21:29:48
@author       :wizz
@version      :1.0
'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# for linux server without GUI
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Model_GRU import MyGRU


def model_train(model, train_loader, num_epochs, learning_rate, device):
    model.train()
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    num_steps = len(train_loader)
    LOSS = []

    for epoch in range(num_epochs):
        total_loss = 0
        # train
        for idx, (inputs, labels) in enumerate(train_loader):

            # 每个样本训练之前先初始化hiddenstate
            model.hidden_cell = torch.zeros(
                model.num_layers, model.batch_size, model.hidden_size)
            inputs = inputs.to(device)
            # ps:将每个BATCH的labels平铺为一维数组！
            labels = labels.view(model.batch_size, -1)
            labels = labels.to(device)

            # forward
            y_out = model(inputs)
            loss = loss_func(y_out, labels)
            total_loss += loss.item()

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (idx + 1) % 1000 == 0:
                print('Epoch: [{}/{}], Step: [{}/{}], Loss: {}'
                      .format(epoch + 1, num_epochs, idx + 1, num_steps, loss.item()))

        LOSS.append(total_loss/num_steps)
        if (epoch + 1) % 5 == 0:
            # 保存模型到文件
            torch.save(model.state_dict(),
                       './model/Epoch' + '%d' % (epoch + 1) + '_GRU.pth')
            print('Save model successfully !')
            # 绘制训练过程损失变化曲线图
            plt.title('Train loss on GRU')
            plt.xlabel('epoch')
            plt.ylabel('MSE')
            plt.grid(True)
            plt.plot(np.arange(1, len(LOSS) + 1),
                     np.array(LOSS), 'k')
            plt.savefig('./train/train_loss.png', format='png', dpi=200)
            plt.close()
