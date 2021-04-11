import pandas as pd
import numpy as np
from datetime import datetime
import random
import sys

import torch
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")

class TextCNN(nn.Module):
    def __init__(self, vector_size, max_seq_len, out_channels):
        super(TextCNN, self).__init__()

        in_channels = 1
        self.kernel_size_list = [4, 2, 1]

        ### 不用torch.nn.Embedding层
        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, vector_size)),
            nn.ReLU(),
            # 经过卷积之后，得到一个维度为max_seq_len - kernel_size + 1的一维向量
            nn.MaxPool2d((max_seq_len - kernel_size + 1, 1))
        ) for kernel_size in self.kernel_size_list])

    def forward(self, input):
        input = torch.unsqueeze(input, 1)
        # Conv2d的输入是个四维的tensor，每一位分别代表(batch_size, channel, length, width)
        batch_size = input.size(0)  # input.size(0)，表示的是输入input的batch_size
        input = [conv(input) for conv in self.convs]
        input = torch.cat(input, dim=1) #按维数1（列）拼接
        input = input.view(batch_size, -1)  # 设经过max pooling之后，有out_channels个数，将x的形状变成(batch_size, out_channels * len(kernel_size_list))，-1表示自适应

        return input

class unilog(torch.nn.Module):
    def __init__(self, vector_size, max_seq_len, out_channels, bilstm_width, bilstm_depth, dropout, device):
        super(unilog, self).__init__()
        
        # [TextCNN, BiLSTM]
        self.upstream_models = nn.ModuleList([
        TextCNN(
            vector_size,
            max_seq_len,
            out_channels
        ),
        torch.nn.LSTM(
            input_size=vector_size,
            hidden_size=bilstm_width,
            num_layers=bilstm_depth,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=True ###控制LSTM是单向的还是双向的，对应PyTorch文档中的num_directions=1或2
        )])

        self.downstream_models = nn.ModuleList([
        nn.Sequential(
            # dropout操作，防止过拟合
            nn.Dropout(dropout),
            # 全连接层，二分类
            ### 双向LSTM，2 * bilstm_width，要乘以2
            nn.Linear(out_channels * len(self.upstream_models[0].kernel_size_list) + (2 if self.upstream_models[1].bidirectional else 1) * bilstm_width, \
                out_channels * len(self.upstream_models[0].kernel_size_list) + (2 if self.upstream_models[1].bidirectional else 1) * bilstm_width),
            nn.ReLU(),
            # dropout操作，防止过拟合
            nn.Dropout(dropout),
            nn.Linear(out_channels * len(self.upstream_models[0].kernel_size_list) + (2 if self.upstream_models[1].bidirectional else 1) * bilstm_width, 2)
        )])

        # 分类
        # self.sm = nn.Softmax(0)

        self.device = device
    
    ### inputs_for_textcnn的格式为(batch_size, max_seq_len, vector_size)
    ### inputs_for_bilstm的格式为(window_size, batch_size, vector_size)
    def forward(self, inputs):
        inputs_for_textcnn = inputs["inputs_for_textcnn"] # inputs.view(inputs.size(0),inputs.size(1)*inputs.size(2),inputs.size(3))
        inputs_for_bilstm = inputs["inputs_for_bilstm"] # torch.sum(inputs, 2)

        output_for_TextCNN = self.upstream_models[0](inputs_for_textcnn)

        h_c = self.get_h_0_c_0(LSTM_model=self.upstream_models[1], seq_len=inputs_for_bilstm.size(0))
        output_for_BiLSTM, _ = self.upstream_models[1](inputs_for_bilstm, h_c)

        output = torch.cat([output_for_TextCNN, output_for_BiLSTM[-1, :, :]] , 1)
        ### output_for_BiLSTM保留最后一个hidden layer的所有神经元的输出
        output = self.downstream_models[0](output)
        
        return output

    def get_h_0_c_0(self, LSTM_model, seq_len):
        if LSTM_model.bidirectional:
            return (torch.zeros(LSTM_model.num_layers*2, seq_len, LSTM_model.hidden_size).to(self.device), \
                torch.zeros(LSTM_model.num_layers*2, seq_len, LSTM_model.hidden_size).to(self.device))
        else:
            return (torch.zeros(LSTM_model.num_layers, seq_len, LSTM_model.hidden_size).to(self.device), \
                torch.zeros(LSTM_model.num_layers, seq_len, LSTM_model.hidden_size).to(self.device))
