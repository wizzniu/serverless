#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file         :dataload.py
@description  :
@time         :2020/11/27 22:22:31
@author       :wizz
@version      :1.0
'''

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self, data, time_step=5, pred_step=5, TRAIN=True):
        self.data = data
        self.time_step = time_step
        self.pred_step = pred_step
        self.TRAIN = TRAIN

    def __getitem__(self, index):
        # trainset
        if self.TRAIN:
            sample = self.data[index: index + self.time_step]
            label = self.data[index + self.time_step: index +
                              self.time_step + self.pred_step]
            # 选择待预测的列作为label，即CPU_USAGE和LAUNCHING_JOB_NUMS
            label = torch.index_select(
                label, dim=1, index=torch.tensor([2, 4]))
            return sample, label
        # testset
        sample = self.data[index*self.time_step: (index + 1)*self.time_step]
        return sample

    def __len__(self):
        if self.TRAIN:
            if (len(self.data) - self.time_step - self.pred_step + 1) >= 0:
                return len(self.data) - self.time_step - self.pred_step + 1
            return 0
        return len(self.data)//self.time_step
