#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file         :Test.py
@description  :
@time         :2020/11/22 23:14:15
@author       :wizz
@version      :1.0
'''


import pandas as pd
import torch
import numpy as np
# for linux server without GUI
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Model_GRU import GRU
from DataLoad import test_dataset, INPUT_SIZE, PRED_STEP


EPOCH_TEST = 5*1
NUM_LAYERS = 2

# load the model paras
model = GRU(input_size=INPUT_SIZE, hidden_size=12,
            num_layers=NUM_LAYERS, output_size=PRED_STEP*2)
model.load_state_dict(torch.load('./model/' + '%d' % EPOCH_TEST + '_GRU.pth'))
model.eval()

actual_pred = []
# prediction
for index, inputs in enumerate(test_dataset):
    with torch.no_grad():
        # 注：这里预测时hidden_cell初始化采用了全零，对预测结果会有影响
        model.hidden_cell = torch.zeros(NUM_LAYERS, 1, model.hidden_size)
        prediction = model(inputs).numpy().astype(int).tolist()
        prediction.insert(0, index + 1)
        actual_pred.append(prediction)


# # inverse_normalize to actual predictions
# actual_pred = np.array(actual_pred).reshape(-1, 1).tolist()
# for i in range(len(actual_pred)):
#     for j in range(INPUT_SIZE - 1):
#         actual_pred[i].append(0)
# actual_pred = scaler.inverse_transform(
#     np.array(actual_pred).reshape(-1, INPUT_SIZE)).tolist()
# for i in range(len(actual_pred)):
#     for j in range(INPUT_SIZE - 1):
#         actual_pred[i].pop()
# actual_pred = np.array(actual_pred).reshape(-1)
# actual_pred[actual_pred < 0] = 0


# evaluate in RMSE
def rmse(predictions, target):
    return np.sqrt(np.square(predictions - target).mean())


# evaluate in MAE
def mae(predictions, target):
    return np.abs(predictions - target).mean()


if __name__ == '__main__':
    df = pd.DataFrame(actual_pred,  columns=['ID', 'CPU_USAGE_1', 'LAUNCHING_JOB_NUMS_1', 'CPU_USAGE_2', 'LAUNCHING_JOB_NUMS_2',
                                             'CPU_USAGE_3', 'LAUNCHING_JOB_NUMS_3', 'CPU_USAGE_4', 'LAUNCHING_JOB_NUMS_4', 'CPU_USAGE_5', 'LAUNCHING_JOB_NUMS_5'])
    # 不保留index
    df.to_csv('./data/result.csv', index=False, encoding='utf-8')
    df.info()
    print(df.head(10))
