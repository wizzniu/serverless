#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file         :main.py
@description  :
@time         :2020/11/28 21:16:20
@author       :wizz
@version      :1.0
'''

import datetime
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from preprocess import data_clean, missing_fix, data_encoding, feature_normalize
from dataload import MyDataSet
from Model_GRU import MyGRU
from train import model_train
from test import model_test


INPUT_SIZE = 10
TIME_STEP = 5
BATCH_SIZE = 64
PRED_STEP = 5
TRAIN_EPOCH = 50
LR = 0.001
TEST_EPOCH = 50


if __name__ == '__main__':
    # 读入训练数据并作预处理
    train = pd.read_csv('./data/raw/train.csv', header=0,
                        parse_dates=['DOTTING_TIME'])
    train = data_clean(train)
    train = missing_fix(train)
    train, le_cu, le_queuetype = data_encoding(train)
    train, scaler_1, scaler_launch = feature_normalize(train)
    # 将处理过的数据样例demo打印到文件中以便查看
    train.head(10).to_csv('./data/result/train_demo.csv',
                          index=False, encoding='utf-8')
    queue_id_list = train['QUEUE_ID'].unique()

    trainset = MyDataSet([])
    # 按QUEUE_ID分组进行训练数据集制作
    for queue_id in tqdm(queue_id_list):
        group = train[train['QUEUE_ID'] == queue_id]
        # 删除QUEUE_ID列
        group = group.drop(columns=['QUEUE_ID'])
        # convert ndarray into tensor
        group = torch.FloatTensor(group.values)
        groupset = MyDataSet(group, TIME_STEP, PRED_STEP)
        # 将分组制作的数据集拼接
        trainset = trainset + groupset
    # 加载数据集
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    # for data, label in train_loader:
    #     print(data)
    #     print(label)
    #     break

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 初始化模型并训练
    model = MyGRU(input_size=INPUT_SIZE, hidden_size=12, num_layers=2,
                  output_size=PRED_STEP*2, batch_size=BATCH_SIZE, drop_out=0.5)
    model.to(device)
    start = datetime.datetime.now()
    print('{}   Start training...'.format(start))
    model_train(model, train_loader, TRAIN_EPOCH, LR, device)
    end = datetime.datetime.now()
    print('Training costs time: {}'.format(end - start))
    print('train model completely !!!')

    # 读入测试数据并作预处理
    test = pd.read_csv('./data/raw/evaluation_public.csv', header=0,
                       parse_dates=['DOTTING_TIME'])
    test = data_clean(test, TRAIN=False)
    test['CU'] = le_cu.transform(test['CU'])
    test['QUEUE_TYPE'] = le_queuetype.transform(test['QUEUE_TYPE'])
    COL_1 = ['CU', 'RUNNING_JOB_NUMS', 'SUCCEED_JOB_NUMS',
             'CANCELLED_JOB_NUMS', 'FAILED_JOB_NUMS']
    test[COL_1] = scaler_1.transform(test[COL_1])
    test[['LAUNCHING_JOB_NUMS']] = scaler_launch.transform(
        test[['LAUNCHING_JOB_NUMS']])
    # 对CPU_USAGE,MEM_USAGE,DISK_USAGE作归一化（/100即可）
    COL_2 = ['CPU_USAGE', 'MEM_USAGE', 'DISK_USAGE']
    test[COL_2] = test[COL_2]/100
    # 删除多余列
    test = test.drop(columns=['ID', 'QUEUE_ID'])
    # 将处理过的数据样例demo打印到文件中以便查看
    test.head(10).to_csv('./data/result/test_demo.csv',
                         index=False, encoding='utf-8')

    # 测试数据集制作
    test = torch.FloatTensor(test.values)
    testset = MyDataSet(test, TIME_STEP, PRED_STEP, TRAIN=False)
    # 加载数据集
    test_loader = DataLoader(testset)

    # 加载模型并测试
    model = MyGRU(input_size=INPUT_SIZE, hidden_size=12, num_layers=2,
                  output_size=PRED_STEP*2, batch_size=1)
    model.load_state_dict(torch.load(
        './model/Epoch' + '%d' % TEST_EPOCH + '_GRU.pth'))
    model.to(device)
    result = model_test(model, test_loader, device)
    # 将预测结果反变换
    result = pd.DataFrame(result, columns=['ID', 'CPU_USAGE_1', 'LAUNCHING_JOB_NUMS_1', 'CPU_USAGE_2', 'LAUNCHING_JOB_NUMS_2',
                                           'CPU_USAGE_3', 'LAUNCHING_JOB_NUMS_3', 'CPU_USAGE_4', 'LAUNCHING_JOB_NUMS_4',
                                           'CPU_USAGE_5', 'LAUNCHING_JOB_NUMS_5'])
    COL_CPU = ['CPU_USAGE_1', 'CPU_USAGE_2',
               'CPU_USAGE_3', 'CPU_USAGE_4', 'CPU_USAGE_5']
    COL_LAUNCH = ['LAUNCHING_JOB_NUMS_1', 'LAUNCHING_JOB_NUMS_2',
                  'LAUNCHING_JOB_NUMS_3', 'LAUNCHING_JOB_NUMS_4', 'LAUNCHING_JOB_NUMS_5']
    result[COL_CPU] = result[COL_CPU]*100
    for col in COL_LAUNCH:
        result[[col]] = scaler_launch.inverse_transform(result[[col]])
    result = result.astype('int')
    result[result[COL_CPU] < 0] = 0
    result[result[COL_CPU] > 100] = 100
    result[result[COL_LAUNCH] < 0] = 0
    # 不保留index
    result.to_csv('./data/result/result.csv', index=False, encoding='utf-8')
