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
# from DataLoad import TIME_STEP, BATCH_SIZE, INPUT_SIZE


INPUT_SIZE = 11
TIME_STEP = 5
BATCH_SIZE = 1


class GRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_layers=1, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=0.5)
        #self.linear1 = nn.Linear(hidden_size, 3)
        #self.linear2 = nn.Linear(3, 6)
        self.fully = nn.Linear(hidden_size, output_size)
        self.hidden_cell = torch.zeros(num_layers, 1, self.hidden_size)

    def forward(self, input):
        # input(time_step,batch_size,input_size)
        # gru_out(time_step,batch_size,hidden_size)
        # output(time_step,batch_size,output_size)
        input_seq = input.view(TIME_STEP, BATCH_SIZE,
                               INPUT_SIZE)  # reshape the input!!!
        gru_out, self.hidden_cell = self.gru(input_seq, self.hidden_cell)
        #linear1_out = self.linear1(gru_out)
        #linear2_out = self.linear2(linear1_out)
        output = nn.functional.relu(self.fully(gru_out))
        # 仅取timestep的最后一步，将输出展开为一维的数组！一维数组的大小=pred_counts*pred_step
        return torch.round(output[-1].view(-1))
