#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file         :Model_GRU.py
@description  :
@time         :2020/11/22 23:11:26
@author       :wizz
@version      :1.0
'''

import torch
import torch.nn as nn


class MyGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_layers=1, output_size=1, batch_size=1, drop_out=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.hidden_cell = torch.zeros(
            num_layers, batch_size, hidden_size)
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=False, dropout=drop_out)
        #self.linear1 = nn.Linear(hidden_size, 3)
        #self.linear2 = nn.Linear(3, 6)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq):
        '''
        input_seq(time_step,batch_size,input_size)
        gru_out(time_step,batch_size,hidden_size)

        '''
        # input_seq: (batch_size,time_step,input_size) -> (time_step,batch_size,input_size)!!!
        input_seq = input_seq.permute(1, 0, 2)
        gru_out, self.hidden_cell = self.gru(input_seq, self.hidden_cell)
        # 仅取timestep的最后一步, size of output：[batch_size, hidden_size]
        output = gru_out[-1, :, :]
        #linear1_out = self.linear1(output)
        #linear2_out = self.linear2(linear1_out)
        output = self.fc(output)

        return output
