﻿import pandas as pd
import numpy as np
from datetime import datetime
import random
import torch
import sys

import warnings
warnings.filterwarnings("ignore")

import torch.nn as nn
class raw_TextCNN(nn.Module):
    def __init__(self,args):
        super(raw_TextCNN, self).__init__()
        vector_size = args['vector_size']
        sentence_len = args['sentence_len']
        n_class = args['n_class']

        in_channels = 1
        out_channels = 100
        kernel_size_list = [4, 2, 1]

        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, vector_size)),
            nn.ReLU(),
            nn.MaxPool2d((sentence_len - kernel_size + 1, 1))
        ) for kernel_size in kernel_size_list])
        self.fc = nn.Linear(out_channels * len(kernel_size_list), n_class)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        batch_size = x.size(0)
        x = [conv(x) for conv in self.convs]
        x = torch.cat(x, dim=1)
        x = x.view(batch_size, -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from sklearn import metrics
import matplotlib.pyplot as plt
class TextCNN(object):
    def __init__(self, log_name, vector_size, sentence_len, n_class):
        self.TextCNN_file_dir = 'TextCNN_file/' + log_name + '/'
        self.vector_size = vector_size
        self.sentence_len = sentence_len
        self.n_class = n_class
        self.batch_size = 10000
        args = {
            'vector_size': vector_size,
            'sentence_len': sentence_len,
            'n_class': n_class
        }
        self.TextCNN_model = raw_TextCNN(args)
    
    def data_process(self, input):
        print('Convert sentences to a fixed length')
        progress_bar = progressbar.ProgressBar(max_value=len(input)) 
        for i in range(len(input)):
            original_len = len(input[i])
            if original_len < self.sentence_len:
                for j in range(self.sentence_len - original_len):
                    input[i].append(np.zeros(self.vector_size))
            else:
                for j in range(original_len - self.sentence_len):
                    del input[i][-1]
            progress_bar.update(i + 1)
        print('\n')
        return input

    def train(self, input, label, do_balance, batch_size, num_epoches):
        if do_balance == True:
            print('Starting doing balance')
            input = np.array(input)
            label = np.array(label)
            positive_num = label.sum()
            negative_num = label.shape[0] - label.sum()
            if positive_num != negative_num:
                if positive_num > negative_num:
                    decreased_label = 1
                    decreased_num = positive_num - negative_num
                else:
                    decreased_label = 0
                    decreased_num = negative_num - positive_num
                decreased_index = np.where(label == decreased_label)[0]
                np.random.shuffle(decreased_index)
                decreased_index = decreased_index[:int(decreased_num)]
                input = np.delete(input, decreased_index.tolist(), axis=0)
                label = np.delete(label, decreased_index.tolist(), axis=0)
            positive_ratio = label.sum() / label.shape[0]
            print('The new positive sample ratio: %.4f after balancing' % (positive_ratio))
            input = input.tolist()
            label = label.tolist()

        input = self.data_process(input)
        dataset = TensorDataset(torch.Tensor(input), torch.Tensor(label))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.TextCNN_model.parameters(), lr=0.1, momentum=0.9)
        print('Start training')
        self.TextCNN_model.train()
        for epoch in range(num_epoches):
            print('Epoch [{}/{}]: '.format(epoch + 1, num_epoches))
            progress_bar = progressbar.ProgressBar(max_value=len(dataloader))
            train_loss = 0
            for step, (input, label) in enumerate(dataloader):
                output = self.TextCNN_model(input)
                loss = criterion(output, label.long())

                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                progress_bar.update(step+1)
            print('\n')

            print('Train_loss: {:.6f}'.format(train_loss / len(dataloader.dataset) * 10**(5)))

        torch.save(self.TextCNN_model.state_dict(), self.TextCNN_file_dir + 'EventTemplate_TextCNN.model')

    def load(self):
        self.TextCNN_model.load_state_dict(torch.load(self.TextCNN_file_dir + 'EventTemplate_TextCNN.model'))

    def evaluate(self, input, label, do_balance):
        if do_balance == True:
            print('Starting doing balance')
            input = np.array(input)
            label = np.array(label)
            positive_num = label.sum()
            negative_num = label.shape[0] - label.sum()
            if positive_num != negative_num:
                if positive_num > negative_num:
                    decreased_label = 1
                    decreased_num = positive_num - negative_num
                else:
                    decreased_label = 0
                    decreased_num = negative_num - positive_num
                decreased_index = np.where(label == decreased_label)[0]
                np.random.shuffle(decreased_index)
                decreased_index = decreased_index[:int(decreased_num)]
                input = np.delete(input, decreased_index.tolist(), axis=0)
                label = np.delete(label, decreased_index.tolist(), axis=0)
            positive_ratio = label.sum() / label.shape[0]
            print('The new positive sample ratio: %.4f after balancing' % (positive_ratio))
            input = input.tolist()
            label = label.tolist()

        input = self.data_process(input)
        dataset = TensorDataset(torch.Tensor(input), torch.Tensor(label))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, drop_last=False)

        print('Start evaluating')
        self.TextCNN_model.eval()
        starttime = datetime.now()
        output = []
        label = []
        progress_bar = progressbar.ProgressBar(max_value=len(dataloader)) 
        for step, (input_temp, label_temp) in enumerate(dataloader):
            output_temp = self.TextCNN_model(input_temp)
            output_temp = F.softmax(output_temp)
            output_temp = output_temp.detach().numpy()
            output_temp = np.argmax(output_temp, axis=1)
            output += output_temp.tolist()
            label += label_temp.detach().numpy().tolist()
            progress_bar.update(step + 1)
        print('\n')
        output = np.array(output)
        label = np.array(label)
        P = metrics.precision_score(label, output)
        R = metrics.recall_score(label, output)
        F1 = metrics.f1_score(label, output)
        confusion_matrix = metrics.confusion_matrix(label, output)
        FPR = confusion_matrix[0, 1] / (np.sum(confusion_matrix[0, :]))
        AUC = metrics.roc_auc_score(label, output)
        print('P: %.6f, R: %.6f, F1: %.6f, FPR: %.6f, AUC: %.6f' % (P, R, F1, FPR, AUC))
        print('Evaluating done (Time taken: {!s})'.format(datetime.now() - starttime))

    def predict(self, input, label):
        input = self.data_process(input)
        dataset = TensorDataset(torch.Tensor(input), torch.Tensor(label).squeeze(1))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, drop_last=False)

        print('Start predicting')
        self.TextCNN_model.eval()
        starttime = datetime.now()
        output_fc = []
        output = []
        label = []
        progress_bar = progressbar.ProgressBar(max_value=len(dataloader)) 
        for step, (input_temp, label_temp) in enumerate(dataloader):
            output_temp = self.TextCNN_model(input_temp)
            output_fc += output_temp.detach().numpy().tolist()
            output_temp = F.softmax(output_temp)
            output_temp = output_temp.detach().numpy()
            output_temp = np.argmax(output_temp, axis=1)
            output += output_temp.tolist()
            label += label_temp.detach().numpy().tolist()
            progress_bar.update(step + 1)
        print('\n')
        output = np.array(output)
        label = np.array(label)
        P = metrics.precision_score(label, output)
        R = metrics.recall_score(label, output)
        F1 = metrics.f1_score(label, output)
        confusion_matrix = metrics.confusion_matrix(label, output)
        FPR = confusion_matrix[0, 1] / (np.sum(confusion_matrix[0, :]))
        AUC = metrics.roc_auc_score(label, output)
        print('P: %.6f, R: %.6f, F1: %.6f, FPR: %.6f, AUC: %.6f' % (P, R, F1, FPR, AUC))
        print('Predicting done (Time taken: {!s})'.format(datetime.now() - starttime))
        return output_fc