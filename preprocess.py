#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file         :preprocess.py
@description  :
@time         :2020/11/27 16:29:12
@author       :wizz
@version      :1.0
'''

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


# 数据清洗
def data_clean(data, TRAIN=True):
    df = data.copy()
    # 将Unix时间戳转化为通用格式
    df['DOTTING_TIME'] = pd.to_datetime(df['DOTTING_TIME'], unit='ms')
    if TRAIN:
        # 按照队列ID和采样时间升序排列
        df = df.sort_values(['QUEUE_ID', 'DOTTING_TIME'])
        df = df[df['STATUS'] == 'available']
        df = df[df['PLATFORM'] == 'x86_64']
        df = df[df['RESOURCE_TYPE'] == 'vm']
        # 删除测试集未出现的QUEUE_TYPE行
        df = df[df['QUEUE_TYPE'] != 'spark']
    # 删除测试集中仅出现单一值的列以及时间列
    df = df.drop(columns=['DOTTING_TIME', 'STATUS',
                          'PLATFORM', 'RESOURCE_TYPE'])
    return df


# 缺失值填充
def missing_fix(data):
    df = data.copy()
    # 按QUEUE_ID进行分组，对'DISK_USAGE'列按分组进行插值填充
    df = df.groupby('QUEUE_ID', group_keys=False).apply(
        lambda group: group.interpolate(method='linear').bfill()).reset_index(drop=True)
    df['DISK_USAGE'] = df['DISK_USAGE'].astype(int)
    return df


# 哑编码与标签编码
def data_encoding(data):
    df = data.copy()
    # 对CU进行标签编码
    le_1 = LabelEncoder()
    le_1 = le_1.fit(df['CU'].sort_values())
    df['CU'] = le_1.transform(df['CU'])

    # # 对QUEUE_TYPE进行哑编码 (剔除第一类的列，drop_first=True)
    # QUEUE_TYPE_Dummy = pd.get_dummies(
    #     df['QUEUE_TYPE'], drop_first=True, prefix='QUEUE_TYPE')
    # # 将原始变量与虚拟变量拼接
    # df = pd.concat([df, QUEUE_TYPE_Dummy], axis=1)
    # df = df.drop(columns='QUEUE_TYPE')

    # 对QUEUE_TYPE也进行标签编码,因其只有两种值（general,sql),效果与哑编码一致
    le_2 = LabelEncoder()
    le_2 = le_2.fit(df['QUEUE_TYPE'])
    df['QUEUE_TYPE'] = le_2.transform(df['QUEUE_TYPE'])
    return df, le_1, le_2


# 归一化
def feature_normalize(data):
    df = data.copy()
    # 对LAUNCHING,RUNNING,SUCCEED,CANCELLED,FAILED_JOB_NUMS和CU作离差标准化
    COL_1 = ['CU', 'RUNNING_JOB_NUMS', 'SUCCEED_JOB_NUMS',
             'CANCELLED_JOB_NUMS', 'FAILED_JOB_NUMS']
    scaler_1 = MinMaxScaler(feature_range=(0, 1))
    scaler_1 = scaler_1.fit(df[COL_1])
    df[COL_1] = scaler_1.transform(df[COL_1])
    # 因LAUNCHING_JOB_NUMS后续要作反标准化，故单独处理
    scaler_2 = MinMaxScaler(feature_range=(0, 1))
    scaler_2 = scaler_1.fit(df[['LAUNCHING_JOB_NUMS']])
    df[['LAUNCHING_JOB_NUMS']] = scaler_2.transform(df[['LAUNCHING_JOB_NUMS']])
    # 对CPU_USAGE,MEM_USAGE,DISK_USAGE作归一化（/100即可）
    COL_2 = ['CPU_USAGE', 'MEM_USAGE', 'DISK_USAGE']
    df[COL_2] = df[COL_2]/100
    return df, scaler_1, scaler_2
