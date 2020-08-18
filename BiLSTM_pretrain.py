import pandas as pd
import numpy as np
from datetime import datetime
import random
import torch
import sys

import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from datetime import datetime
import progressbar
from sklearn import metrics
class raw_BiLSTM(torch.nn.Module):
    def __init__(self, vector_size, hidden_size, num_layers, depth):
        super(raw_BiLSTM, self).__init__()
        self.depth = depth
        LSTM_model_left_to_right_list = []
        fc_left_to_right_list = []
        LSTM_model_right_to_left_list = []
        fc_right_to_left_list = []

        for i in range(self.depth):
            LSTM_model_left_to_right_list.append(torch.nn.LSTM(
                input_size=vector_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=True,
                batch_first=True,
                dropout=0,
                bidirectional=False
            ))
            fc_left_to_right_list.append(torch.nn.Linear(hidden_size, vector_size))
            LSTM_model_right_to_left_list.append(torch.nn.LSTM(
                input_size=vector_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=True,
                batch_first=True,
                dropout=0,
                bidirectional=False
            ))
            fc_right_to_left_list.append(torch.nn.Linear(hidden_size, vector_size))
        self.LSTM_model_left_to_right = torch.nn.ModuleList(LSTM_model_left_to_right_list)
        self.fc_left_to_right = torch.nn.ModuleList(fc_left_to_right_list)
        self.LSTM_model_right_to_left = torch.nn.ModuleList(LSTM_model_right_to_left_list)
        self.fc_right_to_left = torch.nn.ModuleList(fc_right_to_left_list)

    def forward(self, input):
        window_size = input.size(1)
        output = []

        h_c = []
        for j in range(self.depth):
            h_c.append(self.get_h_0_c_0(LSTM_model=self.LSTM_model_left_to_right[j], batch_size = input.size(0)))
        for i in range(window_size - 1):
            vector = input[:, i, :]
            for j in range(self.depth):
                vector, h_c[j] = self.LSTM_model_left_to_right[j](vector.unsqueeze(1), h_c[j]) 
                vector = self.fc_left_to_right[j](vector[:, -1, :])
            output.append(vector)

        h_c = []
        for j in range(self.depth):
            h_c.append(self.get_h_0_c_0(LSTM_model=self.LSTM_model_right_to_left[j], batch_size = input.size(0)))
        for i in range(window_size - 1):
            vector = input[:, -(i+1), :]
            for j in range(self.depth):
                vector, h_c[j] = self.LSTM_model_right_to_left[j](vector.unsqueeze(1), h_c[j]) 
                vector = self.fc_right_to_left[j](vector[:, -1, :])
            output.append(vector)

        return torch.stack(tuple(output), dim=1)

    def get_h_0_c_0(self, LSTM_model, batch_size):
        if LSTM_model.bidirectional:
            return (torch.zeros(LSTM_model.num_layers*2, batch_size, LSTM_model.hidden_size), \
                torch.zeros(LSTM_model.num_layers*2, batch_size, LSTM_model.hidden_size))
        else:
            return (torch.zeros(LSTM_model.num_layers, batch_size, LSTM_model.hidden_size), \
                torch.zeros(LSTM_model.num_layers, batch_size, LSTM_model.hidden_size))

class BiLSTM(object):
    def __init__(self, log_name, vector_size, hidden_size, num_layers, window_size, depth):
        self.BiLSTM_model = raw_BiLSTM(vector_size=vector_size, hidden_size=hidden_size, num_layers=num_layers, depth=depth)
        self.window_size = window_size
        self.BiLSTM_file_dir = 'BiLSTM_file/' + log_name + '/'
        self.batch_size = 10000

    def data_process(self, raw_input, raw_label):
        input = []
        target = []
        label = []
        print('Change data from raw shape to BiLSTM shape: ')
        dataset_size = 0
        for data_of_one_facility in raw_input:
            dataset_size += max(len(data_of_one_facility) - self.window_size  + 1, 0)
        progress_bar = progressbar.ProgressBar(max_value=dataset_size) 
        count = 0
        for i in range(len(raw_input)):
            if len(raw_input[i]) >= self.window_size:
                for j in range(len(raw_input[i]) - self.window_size + 1):
                    input.append(raw_input[i][j:(j + self.window_size)])
                    left_to_right_target = raw_input[i][(j + 1):(j + self.window_size)]
                    right_to_left_target = raw_input[i][j:(j + self.window_size - 1)]
                    right_to_left_target.reverse()
                    target.append(left_to_right_target + right_to_left_target)
                    label.append(raw_label[i][j + self.window_size - 1])
                    count += 1
                    progress_bar.update(count)
        print('\n')

        return input, target, label

    def train(self, input, label, batch_size, num_epoch):
        input, target, label = self.data_process(input, label)
        dataset = TensorDataset(torch.Tensor(input), torch.Tensor(target))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.BiLSTM_model.parameters())
        print('Start training')
        for epoch in range(num_epoch):
            print('Epoch [{}/{}]: '.format(epoch + 1, num_epoch))
            progress_bar = progressbar.ProgressBar(max_value=len(dataloader))
            train_loss = 0
            for step, (input, target) in enumerate(dataloader):
                output = self.BiLSTM_model(input)
                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                progress_bar.update(step+1)
            print('\n')

            print('Train_loss: {:.6f}'.format(train_loss / len(dataloader.dataset) * 100))

        torch.save(self.BiLSTM_model.state_dict(), self.BiLSTM_file_dir + 'BiLSTM_model.model')

    def load_BiLSTM_model(self):
        self.BiLSTM_model.load_state_dict(torch.load(self.BiLSTM_file_dir + 'BiLSTM_model.model'))

    def evaluate(self, input, label):
        input, target, label = self.data_process(input, label)
        
        dataset = TensorDataset(torch.Tensor(input), torch.Tensor(target))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)

        criterion = torch.nn.MSELoss()
        evaluate_loss = 0
        print('Start evaluating')
        starttime = datetime.now()
        progress_bar = progressbar.ProgressBar(max_value=len(dataloader)) 
        for step, (input, target) in enumerate(dataloader):
            output = self.BiLSTM_model(input)
            evaluate_loss += criterion(output, target).item()
            progress_bar.update(step + 1)
        print('\n')
        
        print('Evaluate_loss: {:.6f}'.format(evaluate_loss / len(dataloader.dataset) * 100))
        print('Evaluating done (Time taken: {!s})'.format(datetime.now() - starttime))

    def predict(self, input, label):
        input, target, label = self.data_process(input, label)

        criterion = torch.nn.MSELoss()
        print('Start predicting')
        starttime = datetime.now()

        print('Compute output and label')
        dataset = TensorDataset(torch.Tensor(input), torch.Tensor(label))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)
        output = []
        label = []
        progress_bar = progressbar.ProgressBar(max_value=len(dataloader)) 
        for step, (input_temp, label_temp) in enumerate(dataloader):
            output_temp = self.BiLSTM_model(input_temp)
            output_temp = torch.cat((input_temp.detach(), output_temp.detach()), dim=1)
            output += output_temp.numpy().tolist()
            label += label_temp.detach().numpy().tolist()
            progress_bar.update(step + 1)
        print('\n')
        f = open(self.BiLSTM_file_dir + 'prediction_result.txt', 'w')
        f.write(str(output))
        f.close()
        f = open(self.BiLSTM_file_dir + 'label_for_prediction_result.txt', 'w')
        f.write(str(label))
        f.close()

        print('Compute loss')
        dataset = TensorDataset(torch.Tensor(input), torch.Tensor(target))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)
        predict_loss = 0
        progress_bar = progressbar.ProgressBar(max_value=len(dataloader)) 
        for step, (input_temp, target_temp) in enumerate(dataloader):
            output_temp = self.BiLSTM_model(input_temp)
            predict_loss += criterion(output_temp, target_temp).item()
            progress_bar.update(step + 1)
        print('\n')
        print('Predict_loss: {:.6f}'.format(predict_loss / len(dataloader.dataset) * 100))
        
        print('Predicting done (Time taken: {!s})'.format(datetime.now() - starttime))