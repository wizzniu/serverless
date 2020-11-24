#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file         :DataLoad.py
@description  :
@time         :2020/11/22 23:11:00
@author       :wizz
@version      :1.0
'''


import pandas as pd
import numpy as np
# for linux server without GUI
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
# from sklearn.preprocessing import MinMaxScaler


INPUT_SIZE = 11
TIME_STEP = 5
BATCH_SIZE = 1
PRED_STEP = 5   # PRED_STEP,i.e.,OUTPUT_SIZE
TRAIN_RATE = 0.8


def create_dataset(data, time_step, pred_step):
    dataset = []
    # 一共创建(len(data) - time_step - pred_step + 1)对数据
    for i in range(len(data) - time_step - pred_step + 1):
        input_seq = data[i:i + time_step]
        label_dumplicate = data[i + time_step:i + time_step + pred_step]
        label = []
        for j in range(pred_step):
            label.append([label_dumplicate[j][0], label_dumplicate[j][2]])
        label = torch.FloatTensor(np.array(label))
        dataset.append((input_seq, label))
    return dataset


def create_test_dataset(data, pred_step):
    testset = []
    for i in range(0, len(data), pred_step):
        testset.append(data[i])
    return testset


# without labels
def create_evaluation_dataset(data, time_step):
    evalset = []
    for i in range(int(len(data)/time_step)):
        evalset.append(data[i:i + time_step])
    return evalset


#############  对训练集的处理  ################
# 读取所有数据
all_data = pd.read_csv('./data/train_to_use.csv', header=0)
all_data = all_data.drop(columns=['CU', 'DOTTING_TIME'])
queue_id_list = all_data['QUEUE_ID'].unique()
train_dataset = []
validation_dataset = []
# 按QUEUE_ID分组进行训练集数据集制作
for queue_id in queue_id_list:
    group = all_data[all_data['QUEUE_ID'] == queue_id]
    group = group.drop(columns=['QUEUE_ID'])
    # split data into train_data(80%) and validation_data(20%)
    train_data_size = int(len(group) * TRAIN_RATE)
    train_data = group[:train_data_size]
    validation_data = group[train_data_size:]
    # convert ndarray into tensor
    train_data = torch.FloatTensor(train_data.values)
    validation_data = torch.FloatTensor(validation_data.values)
    # create dataset
    train_dataset_group = create_dataset(train_data, TIME_STEP, PRED_STEP)
    train_dataset.extend(train_dataset_group)
    validation_dataset_duplicate = create_dataset(
        validation_data, TIME_STEP, PRED_STEP)
    validation_dataset_group = create_test_dataset(
        validation_dataset_duplicate, PRED_STEP)
    validation_dataset.extend(validation_dataset_group)

# print(len(train_dataset))
# print(train_dataset[0])
# print(len(validation_dataset))
# print(validation_dataset[0])
################################################


#############  对测试样例的处理  ################
# 读取所有数据
test_data = pd.read_csv('./data/evaluation_to_use.csv', header=0)
test_data = test_data.drop(columns=['ID', 'QUEUE_ID', 'CU', 'DOTTING_TIME'])
# convert ndarray into tensor
test_data = torch.FloatTensor(test_data.values)
test_dataset = create_evaluation_dataset(test_data, TIME_STEP)

################################################


# # normalize (fit only for train_data!!!)
# scaler = MinMaxScaler(feature_range=(0, 1))     # 转换后的数据不覆盖原数据
# scaler = scaler.fit(train_data.reshape(-1, INPUT_SIZE))
# train_data_normalized = scaler.transform(train_data.reshape(-1, INPUT_SIZE))
# #print(train_data_normalized[-(TIME_STEP + PRED_STEP):])
# validation_data_normalized = scaler.transform(
#     validation_data.reshape(-1, INPUT_SIZE))
# #print(validation_data_normalized[-(TIME_STEP + PRED_STEP):])
# test_data_normalized = scaler.transform(test_data.reshape(-1, INPUT_SIZE))
# #print(test_data_normalized[-(TIME_STEP + PRED_STEP):])


if __name__ == '__main__':
    plt.figure(figsize=(13, 5))
    plt.title('all_data')
    plt.xlabel('time&queue_id')
    plt.ylabel('...')
    plt.grid(True)
    plt.plot(all_data.drop(columns=['QUEUE_ID']))
    plt.legend(['CPU_USAGE', 'MEM_USAGE', 'LAUNCHING_JOB_NUMS', 'RUNNING_JOB_NUMS', 'SUCCEED_JOB_NUMS',
                'CANCELLED_JOB_NUMS', 'FAILED_JOB_NUMS', 'DISK_USAGE', 'QUEUE_TYPE_general', 'QUEUE_TYPE_spark', 'QUEUE_TYPE_sql'])
    plt.savefig('./data/train_to_use.png', format='png', dpi=200)
    plt.close()
