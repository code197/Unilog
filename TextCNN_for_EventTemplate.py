import pandas as pd
import numpy as np
from datetime import datetime
import random
import torch
import sys

import warnings
warnings.filterwarnings("ignore")

### 不对文本中的单词进行标号
### 只是建立词表（用集合类型 set）
### text的类型为2维list
### 在训练完毕后，遇到单个的新单词时，用全0的向量表示；在把整句话映射成向量形式时，如果遇到新单词，对词表进行更新，同时更新word2vec模型
from gensim.models import Word2Vec
class Word2Vec_class(object):
    def __init__(self, make_vocabulary, train_model, Word2Vec_file_dir, \
        text, min_count, vector_size, seed, worker_num):
        self.Word2Vec_file_dir = Word2Vec_file_dir
        if make_vocabulary == True:
            print('Start making the vocabulary file')
            starttime = datetime.now()
            self.Word2Vec_file_dir = Word2Vec_file_dir
            self.vocabulary = []

            ###显示进度条
            print('Count words')
            progress_bar = progressbar.ProgressBar(max_value=len(text))
            for i in range(len(text)):
                for word in text[i]:
                    self.vocabulary.append(word)
                progress_bar.update(i + 1)
            print('\n')
            self.vocabulary = set(self.vocabulary)
            self.vocabulary_size = len(self.vocabulary)
            #self.word_to_num = {word: str(i) for i, word in enumerate(self.vocabulary)}
            #self.num_to_word = {self.word_to_num[word]: word for word in self.word_to_num}
            
            print('Making done (Time taken: {!s})'.format(datetime.now() - starttime))
            f = open(self.Word2Vec_file_dir + 'vocabulary.txt', 'w')
            f.write(str(self.vocabulary))
            f.close()
        else:
            f = open(self.Word2Vec_file_dir + 'vocabulary.txt', 'r')
            self.vocabulary = eval(f.read())
            f.close()
            self.vocabulary_size = len(self.vocabulary)

        if train_model == True:
            print('Start training the word2vec model')
            starttime = datetime.now()
            self.Word2Vec_model = Word2Vec(sentences=text, min_count=min_count, size=vector_size, seed=seed, workers=worker_num)
            print('Training done (Time taken: {!s})'.format(datetime.now() - starttime))
            self.Word2Vec_model.save(self.Word2Vec_file_dir + 'EventTemplate_Word2Vec.model')
        else:
            self.Word2Vec_model = Word2Vec.load(self.Word2Vec_file_dir + 'EventTemplate_Word2Vec.model')
        self.vector_size = self.Word2Vec_model.vector_size

    ### word的类型是str；返回1维array
    def get_vector(self, word):
        # Word2Vec输出的向量的模长不等于1
        if word not in self.vocabulary:
            return np.zeros(self.vector_size)
        else:
            return self.Word2Vec_model[self.word_to_num[word]]

    ### text的类型是list->list->str；返回list->list->array
    def words_to_vectors(self, text):
        print('Start updating the vocabulary')
        starttime = datetime.now()
        self.Word2Vec_model.build_vocab(sentences=text, update=True)
        print('Updating done (Time taken: {!s})'.format(datetime.now() - starttime))
        self.Word2Vec_model.save(self.Word2Vec_file_dir + 'EventTemplate_Word2Vec.model')
        
        print('Start converting the words to the vectors')
        progress_bar = progressbar.ProgressBar(max_value=len(text))
        text_vector = []
        for i in range(len(text)):
            text_vector.append([])
            for j in range(len(text[i])):
                text_vector[i].append(self.Word2Vec_model[text[i][j]])
            progress_bar.update(i + 1)
        print('\n')

        return text_vector

def Word2Vec_create(log_name, make_vocabulary, train_model, \
    text, min_count, vector_size, seed, worker_num):
    Word2Vec_file_dir = 'Word2Vec_file/' + log_name + '_EventTemplate_Ours' + '/'
    return Word2Vec_class(make_vocabulary=make_vocabulary, train_model=train_model, Word2Vec_file_dir=Word2Vec_file_dir, \
        text=text, min_count=min_count, vector_size=vector_size, seed=seed, worker_num=worker_num)

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

        ### 不用torch.nn.Embedding层
        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, vector_size)),
            nn.ReLU(),
            # 经过卷积之后，得到一个维度为sentence_len - kernel_size + 1的一维向量
            nn.MaxPool2d((sentence_len - kernel_size + 1, 1))
        ) for kernel_size in kernel_size_list])
        # 全连接层，二分类
        self.fc = nn.Linear(out_channels * len(kernel_size_list), n_class)
        # dropout操作，防止过拟合
        self.dropout = nn.Dropout(0.5)
        # 分类
        #self.sm = nn.Softmax(0)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        # Conv2d的输入是个四维的tensor，每一位分别代表batch_size、channel、length、width
        batch_size = x.size(0)  # x.size(0)，表示的是输入x的batch_size
        x = [conv(x) for conv in self.convs]
        x = torch.cat(x, dim=1) #按维数1（列）拼接
        x = x.view(batch_size, -1)  # 设经过max pooling之后，有out_channels个数，将x的形状变成(batch_size, out_channels)，-1表示自适应
        x = self.dropout(x)
        x = self.fc(x)  # nn.Linear接收的参数类型是二维的tensor(batch_size, out_channels),一批有多少数据，就有多少行
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
        self.batch_size = 10000 ### 用于evaluate和predict的batch size，而不是train的batch size
        args = {
            'vector_size': vector_size,
            'sentence_len': sentence_len,
            'n_class': n_class
        }
        self.TextCNN_model = raw_TextCNN(args)
    
    ### 把句子转化为固定长度
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

    ### input是list->list->array(数据量*句子长度*向量长度)，label是array(数据量)
    def train(self, input, label, do_balance, batch_size, num_epoches):
        ### 对训练数据做平衡
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
        #print('torch.Tensor(input).shape: ', torch.Tensor(input).shape)
        #print('torch.Tensor(label).shape: ', torch.Tensor(label).shape)
        ### 为了解决CrossEntropyLoss()的错误 RuntimeError: 1D target tensor expected, multi-target not supported
        ### torch.Tensor(label).squeeze(1) 把形状由(数据量, 1)变为(数据量,)
        #dataset = TensorDataset(torch.Tensor(input), torch.Tensor(label).squeeze(1))
        dataset = TensorDataset(torch.Tensor(input), torch.Tensor(label))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)

        ###Loss and Optimizer
        criterion = torch.nn.CrossEntropyLoss()
        #optimizer = torch.optim.Adam(self.TextCNN_model.parameters())
        optimizer = torch.optim.SGD(self.TextCNN_model.parameters(), lr=0.1, momentum=0.9)
        print('Start training')
        self.TextCNN_model.train()  ### 很重要，不要忘记！
        for epoch in range(num_epoches):
            print('Epoch [{}/{}]: '.format(epoch + 1, num_epoches))
            progress_bar = progressbar.ProgressBar(max_value=len(dataloader))
            ### len(dataloader.dataset)等于数据集的数据量，len(dataloader)等于数据集的数据量/batch_size
            #print('len(dataloader.dataset): ', len(dataloader.dataset))
            #print('len(dataloader): ', len(dataloader))
            train_loss = 0
            for step, (input, label) in enumerate(dataloader):
                output = self.TextCNN_model(input)
                #print('output.shape: ', output.shape)
                #print('label.shape: ', label.shape)
                ### CrossEntropyLoss()的输入格式
                ### 第1个参数 Input: (N,C) C 是类别的数量
                ### 第2个参数 Target: (N) N是mini-batch的大小，0 <= targets[i] <= C-1
                loss = criterion(output, label.long())
                ### label.long()解决错误 RuntimeError: expected scalar type Long but found Float

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                progress_bar.update(step+1)
                #print('step+1: ', step+1)
            print('\n')

            print('Train_loss: {:.6f}'.format(train_loss / len(dataloader.dataset) * 10**(5)))

        torch.save(self.TextCNN_model.state_dict(), self.TextCNN_file_dir + 'EventTemplate_TextCNN.model')

    def load(self):
        self.TextCNN_model.load_state_dict(torch.load(self.TextCNN_file_dir + 'EventTemplate_TextCNN.model'))

    ### input是list->list->array(数据量*句子长度*向量长度)，label是list(数据量)
    def evaluate(self, input, label, do_balance):
        ### 对测试数据做平衡
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
        #print('torch.Tensor(input).shape: ', torch.Tensor(input).shape)
        #print('torch.Tensor(label).shape: ', torch.Tensor(label).shape)
        #dataset = TensorDataset(torch.Tensor(input), torch.Tensor(label).squeeze(1))
        dataset = TensorDataset(torch.Tensor(input), torch.Tensor(label))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, drop_last=False)

        print('Start evaluating')
        self.TextCNN_model.eval() ### 很重要，不要忘记！
        starttime = datetime.now()
        output = []
        label = []
        progress_bar = progressbar.ProgressBar(max_value=len(dataloader)) 
        for step, (input_temp, label_temp) in enumerate(dataloader):
            output_temp = self.TextCNN_model(input_temp)
            # output.append(output_temp) ### 这种写法并没有减缓内存压力，因为该变量没有销毁，相关数据一直存着
            #print('output.shape: ', output.shape)
            output_temp = F.softmax(output_temp)
            #print('output.shape (after softmax): ', output.shape)
            ### RuntimeError: Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
            output_temp = output_temp.detach().numpy()
            output_temp = np.argmax(output_temp, axis=1) ### np.argmax返回最大值对应的索引，axis=0沿列求最大值，axis=1沿行求最大值
            #print('output.shape (after argmax): ', output.shape)
            ### 必须是 += 而不是 append，注意维度正确
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

    ### input是list->list->array(数据量*句子长度*向量长度)，label是list(数据量)
    def predict(self, input, label):
        '''
        label_ = np.array(label) ### 测试顺序是否会变化
        '''
        input = self.data_process(input)
        #input = torch.Tensor(input)
        #print('len(input): ', len(input))
        #print('len(label): ', len(label))
        dataset = TensorDataset(torch.Tensor(input), torch.Tensor(label).squeeze(1))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, drop_last=False)

        print('Start predicting')
        self.TextCNN_model.eval() ### 很重要，不要忘记！
        starttime = datetime.now()
        output_fc = []
        output = []
        label = []
        progress_bar = progressbar.ProgressBar(max_value=len(dataloader)) 
        for step, (input_temp, label_temp) in enumerate(dataloader):
            output_temp = self.TextCNN_model(input_temp)
            ### 把fc层的输出（softmax层之前），作为整体网络中EventTemplate Classification网络的输出
            output_fc += output_temp.detach().numpy().tolist() ### 从2维Tensor转化为1维list->1维array
            # output.append(output_temp) ### 这种写法并没有减缓内存压力，因为该变量没有销毁，相关数据一直存着（例如梯度数据）
            output_temp = F.softmax(output_temp)
            #print('output.shape (after softmax): ', output.shape)
            ### RuntimeError: Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
            output_temp = output_temp.detach().numpy()
            output_temp = np.argmax(output_temp, axis=1) ### np.argmax返回最大值对应的索引，axis=0沿列求最大值，axis=1沿行求最大值
            #print('output.shape (after argmax): ', output.shape)
            ### 必须是 += 而不是 append，注意维度正确
            output += output_temp.tolist() ### 从1维array转化为1维list
            label += label_temp.detach().numpy().tolist() ### 从1维Tensor转化为1维list
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
        '''
        ### 测试顺序是否会变化
        P = metrics.precision_score(label_, label)
        R = metrics.recall_score(label_, label)
        F1 = metrics.f1_score(label_, label)
        confusion_matrix = metrics.confusion_matrix(label_, label)
        FPR = confusion_matrix[0, 1] / (np.sum(confusion_matrix[0, :]))
        AUC = metrics.roc_auc_score(label_, label)
        print('Test order, P: %.6f, R: %.6f, F1: %.6f, FPR: %.6f, AUC: %.6f' % (P, R, F1, FPR, AUC))
        print('\nPredicting done (Time taken: {!s})'.format(datetime.now() - starttime))
        '''
        return output_fc

def process_all_data(Word2Vec_model, TextCNN_model, input, label):
    input = Word2Vec_model.words_to_vectors(text=input)
    output = TextCNN_model.predict(input, label)
    pd.DataFrame(output, columns=['Normal', 'Anomaly']).to_csv(TextCNN_model.TextCNN_file_dir + 'prediction_result.csv', index=False)