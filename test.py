#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file         :test.py
@description  :
@time         :2020/11/29 00:11:57
@author       :wizz
@version      :1.0
'''

import numpy as np
import torch


def model_test(model, test_loader, device):
    model.eval()
    PREDICT = []
    # prediction
    for idx, inputs in enumerate(test_loader):
        with torch.no_grad():
            # 注：这里预测时hidden_cell初始化采用了全零，对预测结果会有影响
            model.hidden_cell = torch.zeros(
                model.num_layers, model.batch_size, model.hidden_size)
            inputs = inputs.to(device)
            predict = model(inputs).view(-1).numpy().tolist()
            # 插入ID
            predict.insert(0, idx + 1)
            PREDICT.append(predict)
    PREDICT = np.array(PREDICT)
    return PREDICT
