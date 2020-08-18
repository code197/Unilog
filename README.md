# Unilog
## 1. TextCNN_for_EventTemplate.py
### 1.1. Package Requirements
```
torch==1.4.0
numpy==1.18.1
pandas==1.0.1
gensim==3.8.1
progressbar33==2.4
scikit-learn==0.23.2
matplotlib==3.1.3
```
### 1.2. classWord2Vec_class
The class is based on gensim.models.Word2Vec, which is used to train and update word embedding vectors.
### 1.3. class raw_TextCNN
The class is the backbone of our TextCNN model, which is uesd for CNN initialization and forward propagation.
### 1.4. class TextCNN
The class is used to instantiate TextCNN model and train it. 

## 2. BiLSTM_pretrain.py
### 2.1. Package Requirements
```
torch==1.4.0
numpy==1.18.1
pandas==1.0.1
progressbar33==2.4
scikit-learn==0.23.2
```
### 2.2. class raw_BiLSTM
The class is the backbone of our BiLSTM model, which is uesd for BiLSTM initialization and forward propagation.
### 2.3. class BiLSTM
The class is used to instantiate BiLSTM model and train it. 

## 3. TextCNN_for_classification.py
### 3.1. Package Requirements
```
torch==1.4.0
numpy==1.18.1
pandas==1.0.1
progressbar33==2.4
scikit-learn==0.23.2
matplotlib==3.1.3
```
### 3.2. Similar to TextCNN_for_EventTemplate.py
Both of the class raw_TextCNN and the class TextCNN are similar to those in TextCNN_for_EventTemplate.py
