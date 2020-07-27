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
### LSTM训练时，会自动使用所有CPU
### depth是LSTM网络的个数
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
                bidirectional=False ###控制LSTM是单向的还是双向的，对应PyTorch文档中的num_directions=1或2
            ))
            fc_left_to_right_list.append(torch.nn.Linear(hidden_size, vector_size))
            LSTM_model_right_to_left_list.append(torch.nn.LSTM(
                input_size=vector_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=True,
                batch_first=True,
                dropout=0,
                bidirectional=False ###控制LSTM是单向的还是双向的，对应PyTorch文档中的num_directions=1或2
            ))
            fc_right_to_left_list.append(torch.nn.Linear(hidden_size, vector_size))
        '''
        _modules成员起很重要的桥梁作用，在获取一个net的所有的parameters的时候，是通过递归遍历该net的所有_modules来实现的。
        像前述提到的那个问题，如果将这些成员都放倒一个python list里：self.layer1 = [conv1, pool, conv2] ——会导致CivilNet不能将conv1, pool, conv2等划归到_modules里，
        从而通过CivilNet的parameters()获取所有权重参数时，拿到的东西为空，就会报 ValueError: optimizer got an empty parameter list 这样的错误。针对这种情况，那怎么办呢？
        ModuleList就是为了解决这个问题的，首先，ModuleList类的基类正是Module：
        class ModuleList(Module)
        其次，ModuleList实现了python的list的功能；
        最后，在使用ModuleList的时候，该类会使用基类（也就是Module）的add_module()方法，或者直接操作_modules成员来将list中的module成功注册。
        Sequential模块也具备ModuleList这样的注册功能，另外其还实现了forward，这是和ModuleList不同的地方：
        '''
        self.LSTM_model_left_to_right = torch.nn.ModuleList(LSTM_model_left_to_right_list)
        self.fc_left_to_right = torch.nn.ModuleList(fc_left_to_right_list)
        self.LSTM_model_right_to_left = torch.nn.ModuleList(LSTM_model_right_to_left_list)
        self.fc_right_to_left = torch.nn.ModuleList(fc_right_to_left_list)

    ### 对于某一层的LSTM，
    ### 输入为当前迭代的前一层的LSTM的输出vector(t)(i-1)，和前一迭代的当前层的LSTM的状态h_c(t-1)(i)
    ### 输出为vector(t)(i)
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
                ###把h_c作为初始状态，h_c可能是全0，也可能是上一次迭代的状态
                ###数据输入到LSTM中,可能因为切片操作，会缺少一个维度，用.unsqueeze(1)加上
                vector = self.fc_left_to_right[j](vector[:, -1, :]) ### 最后一个hidden layer的所有神经元的输出经过全连接层转化为词向量的维度
            output.append(vector)

        h_c = []
        for j in range(self.depth):
            h_c.append(self.get_h_0_c_0(LSTM_model=self.LSTM_model_right_to_left[j], batch_size = input.size(0)))
        for i in range(window_size - 1):
            vector = input[:, -(i+1), :]
            for j in range(self.depth):
                vector, h_c[j] = self.LSTM_model_right_to_left[j](vector.unsqueeze(1), h_c[j]) 
                ###把h_c作为初始状态，h_c可能是全0，也可能是上一次迭代的状态
                ###数据输入到LSTM中,可能因为切片操作，会缺少一个维度，用.unsqueeze(1)加上
                vector = self.fc_right_to_left[j](vector[:, -1, :]) ### 最后一个hidden layer的所有神经元的输出经过全连接层转化为词向量的维度
            output.append(vector)

        return torch.stack(tuple(output), dim=1)
    
    '''
    ### 同一层LSTM之间h_c不传递，只是输入数据有递增关系
    def forward(self, input):
        output = []
        for i in range(self.width):
            h_c = []
            for j in range(i+1):
                h_c.append(self.get_h_0_c_0(batch_size = input.size(0)))
            for j in range(i+1):
                vector = input[:, j, :]
                for k in range(self.depth)):
                    vector, h_c[k] = self.LSTM_model_left_to_right[i][k](vector.unsqueeze(1), h_c[k]) 
                    ###把h_c作为初始状态，h_c可能是全0，也可能是上一次迭代的状态
                    ###数据输入到LSTM中,可能因为切片操作，会缺少一个维度，用.unsqueeze(1)加上
                    vector = self.fc(vector[:, -1, :]) ### 最后一个hidden layer的所有神经元的输出经过全连接层转化为词向量的维度
            output.append(vector)
        for i in range(self.width):
            h_c = []
            for j in range(i+1):
                h_c.append(self.get_h_0_c_0(batch_size = input.size(0)))
            for j in range(i+1):
                vector = input[:, -(j + 1), :]
                for k in range(self.depth):
                    vector, h_c[k] = self.LSTM_model_right_to_left[i][k](vector.unsqueeze(1), h_c[k]) 
                    ###把h_c作为初始状态，h_c可能是全0，也可能是上一次迭代的状态
                    ###数据输入到LSTM中,可能因为切片操作，会缺少一个维度，用.unsqueeze(1)加上
                    vector = self.fc(vector[:, -1, :]) ### 最后一个hidden layer的所有神经元的输出经过全连接层转化为词向量的维度
            output.append(vector)

        return torch.stack(tuple(output), dim=1)
    '''

    def get_h_0_c_0(self, LSTM_model, batch_size):
        if LSTM_model.bidirectional:
            return (torch.zeros(LSTM_model.num_layers*2, batch_size, LSTM_model.hidden_size), \
                torch.zeros(LSTM_model.num_layers*2, batch_size, LSTM_model.hidden_size))
        else:
            return (torch.zeros(LSTM_model.num_layers, batch_size, LSTM_model.hidden_size), \
                torch.zeros(LSTM_model.num_layers, batch_size, LSTM_model.hidden_size))

### use_str 和 use_num 有且只有一个为True
class BiLSTM(object):
    def __init__(self, log_name, vector_size, hidden_size, num_layers, window_size, depth):
        self.BiLSTM_model = raw_BiLSTM(vector_size=vector_size, hidden_size=hidden_size, num_layers=num_layers, depth=depth)
        self.window_size = window_size
        self.BiLSTM_file_dir = 'BiLSTM_file/' + log_name + '/'
        self.batch_size = 10000 ### 用于evaluate的batch size，而不是train的batch size

    ### 输入raw_input为list(Facility数, 每个Facility的vector数, vector_size), raw_label为list(Facility数, 每个Facility的vector数)
    ### 返回input为list(数据量, self.window_size, vector_size)，返回target为list(数据量, 2*(self.window_size-1), vector_size)
    def data_process(self, raw_input, raw_label):
        input = []
        target = []
        label = []
        print('Change data from raw shape to BiLSTM shape: ')
        dataset_size = 0
        for data_of_one_facility in raw_input:
            dataset_size += max(len(data_of_one_facility) - self.window_size  + 1, 0)
        progress_bar = progressbar.ProgressBar(max_value=dataset_size) 
        ### 实际上每行最多有len(data_of_one_facility) - self.window_size  + 1个窗口
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

    ### 输入input为list(Facility数, 每个Facility的vector数, vector_size), label为list(Facility数, 每个Facility的vector数)
    def train(self, input, label, batch_size, num_epoch):
        input, target, label = self.data_process(input, label)
        #print('torch.Tensor(input).shape: ', torch.Tensor(input).shape)
        #print('torch.Tensor(target).shape: ', torch.Tensor(target).shape)
        #print('torch.Tensor(label).shape: ', torch.Tensor(label).shape)
        ### 把所有正常和异常的数据都用于预训练，可以发现异常数据中的错误累积效应
        ###### 测试！！！
        ###### dataset = TensorDataset(torch.Tensor(input[:10000]), torch.Tensor(target[:10000]))
        dataset = TensorDataset(torch.Tensor(input), torch.Tensor(target))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)

        ###Loss and Optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.BiLSTM_model.parameters())
        print('Start training')
        for epoch in range(num_epoch):
            print('Epoch [{}/{}]: '.format(epoch + 1, num_epoch))
            progress_bar = progressbar.ProgressBar(max_value=len(dataloader))
            ### len(dataloader.dataset)等于数据集的数据量，len(dataloader)等于数据集的数据量/batch_size
            #print('len(dataloader.dataset): ', len(dataloader.dataset))
            #print('len(dataloader): ', len(dataloader))
            train_loss = 0
            for step, (input, target) in enumerate(dataloader):
                output = self.BiLSTM_model(input)
                loss = criterion(output, target)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                progress_bar.update(step+1)
                #print('step+1: ', step+1)
            print('\n')

            print('Train_loss: {:.6f}'.format(train_loss / len(dataloader.dataset) * 100))

        torch.save(self.BiLSTM_model.state_dict(), self.BiLSTM_file_dir + 'BiLSTM_model.model')

    def load_BiLSTM_model(self):
        self.BiLSTM_model.load_state_dict(torch.load(self.BiLSTM_file_dir + 'BiLSTM_model.model'))

    ### 可以设置与label向量的相似度在前 slackness名 就预测正确，或者用建设检验，还没写
    ### 输入input为list(Facility数, 每个Facility的vector数, vector_size), label为list(Facility数, 每个Facility的vector数)
    def evaluate(self, input, label):
        input, target, label = self.data_process(input, label)
        
        ###### 测试！！！
        ###### dataset = TensorDataset(torch.Tensor(input[:10000]), torch.Tensor(target[:10000]))
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

    ### 输入input为list(Facility数, 每个Facility的vector数, vector_size), label为list(Facility数, 每个Facility的vector数)
    def predict(self, input, label):
        input, target, label = self.data_process(input, label)

        criterion = torch.nn.MSELoss()
        print('Start predicting')
        starttime = datetime.now()

        print('Compute output and label')
        ###### 测试！！！
        ###### dataset = TensorDataset(torch.Tensor(input[:10000]), torch.Tensor(label[:10000]))
        dataset = TensorDataset(torch.Tensor(input), torch.Tensor(label))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)
        output = []
        label = []
        progress_bar = progressbar.ProgressBar(max_value=len(dataloader)) 
        for step, (input_temp, label_temp) in enumerate(dataloader):
            output_temp = self.BiLSTM_model(input_temp)
            output_temp = torch.cat((input_temp.detach(), output_temp.detach()), dim=1)
            output += output_temp.numpy().tolist() ### 从3维array转化为3维list
            label += label_temp.detach().numpy().tolist() ### 从1维Tensor转化为1维list
            progress_bar.update(step + 1)
        print('\n')
        f = open(self.BiLSTM_file_dir + 'prediction_result.txt', 'w')
        f.write(str(output))
        f.close()
        f = open(self.BiLSTM_file_dir + 'label_for_prediction_result.txt', 'w')
        f.write(str(label))
        f.close()

        print('Compute loss')
        ###### 测试！！！
        ###### dataset = TensorDataset(torch.Tensor(input[:10000]), torch.Tensor(target[:10000]))
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