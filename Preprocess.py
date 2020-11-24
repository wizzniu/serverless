#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file         :Preprocess.py
@description  :
@time         :2020/11/22 23:10:07
@author       :wizz
@version      :1.0
'''


import pandas as pd


#############  对训练集的处理  ################
df = pd.read_csv('./data/train.csv', header=0, parse_dates=['DOTTING_TIME'])
# 将Unix时间戳转化为通用格式
df['DOTTING_TIME'] = pd.to_datetime(df['DOTTING_TIME'], unit='ms')
# print(df.head(10))
# df.info()
# 按QUEUE_ID进行分组，按DOTTING_TIME进行升序
# 通过设置group_keys参数对multiindex进行优化,group_keys=False可以禁用分组键所形成的索引
df = df.groupby('QUEUE_ID', group_keys=False).apply(
    lambda group: group.sort_values('DOTTING_TIME')).reset_index(drop=True)
# 去除无关的列
df = df.drop(columns=['STATUS', 'PLATFORM', 'RESOURCE_TYPE'])
# 对QUEUE_TYPE进行one-hot编码 (不剔除第一类的列，drop_first=False)
QUEUE_TYPE_Dummy = pd.get_dummies(
    df['QUEUE_TYPE'], drop_first=False, prefix='QUEUE_TYPE')
# 将原始变量与虚拟变量拼接
df = pd.concat([df, QUEUE_TYPE_Dummy], axis=1)
df = df.drop(columns='QUEUE_TYPE')
# 按QUEUE_ID进行分组，对'DISK_USAGE'列按分组进行插值填充
df = df.groupby('QUEUE_ID', group_keys=False).apply(
    lambda group: group.interpolate(method='linear').bfill()).reset_index(drop=True)
df['DISK_USAGE'] = df['DISK_USAGE'].astype(int)

# 不保留index
# df.to_csv('./data/train_to_use.csv', index=False)
# print(df.head(10))
# df.info()
# df = df.head(10000)
# df.to_csv('./data/demo_10000.csv', index=False)
################################################


#############  对测试样例的处理  ################
df2 = pd.read_csv('./data/evaluation_public.csv',
                  header=0, parse_dates=['DOTTING_TIME'])
# 将Unix时间戳转化为通用格式
df2['DOTTING_TIME'] = pd.to_datetime(df2['DOTTING_TIME'], unit='ms')
# 去除无关的列
df2 = df2.drop(columns=['STATUS', 'PLATFORM', 'RESOURCE_TYPE'])
# 对QUEUE_TYPE进行one-hot编码 (不剔除第一类的列，drop_first=False)
QUEUE_TYPE_Dummy = pd.get_dummies(
    df2['QUEUE_TYPE'], drop_first=False, prefix='QUEUE_TYPE')
# 将原始变量与虚拟变量拼接
df2 = pd.concat([df2, QUEUE_TYPE_Dummy], axis=1)
# 由于测试集中未出现该列，故新增全为0的虚拟变量
df2.insert(14, 'QUEUE_TYPE_spark', 0)
df2 = df2.drop(columns='QUEUE_TYPE')
df2['DISK_USAGE'] = df2['DISK_USAGE'].astype(int)

# 不保留index
df2.to_csv('./data/evaluation_to_use.csv', index=False)
# print(df2.head(10))
# df2.info()
################################################
