# -*- coding: utf-8 -*-
"""SELFIE.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18EoFHm7PhtWMiLeEq7AQgIB3J-4v0UGb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib
font = {'size': 14}
matplotlib.rc('font', **font)

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import f1_score, accuracy_score, average_precision_score, precision_score, recall_score
from train import *

num_classes = 7
batch_size = 256
test_batch_size = 100
noise_mode = 'symm'

log_interval = 40
learning_rate = 1e-6
weight_decay = 1e-6
reg_term = 0
restart = 1

epochs = 20
history_length = 10
milestones = [history_length, epochs]

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes):
        # 항상 torch.nn.Module을 상속받고 시작
        super(CNN, self).__init__()
        self.keep_prob = .5
        self.num_classes = num_classes

        # L1 img shape = (?, 26, 26, 3)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2))

        # L2 input shape (?, 12, 12, 32)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2))

        # L3 input shape (?, 6, 6, 64)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.fc1 = nn.Linear(128 * 3 * 3, 625, bias=True)
        nn.init.xavier_uniform_(self.fc1.weight)

        self.layer4 = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(1 - self.keep_prob))

        self.fc2 = nn.Linear(625, num_classes, bias=True)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        # print(x.size())
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)  # Flatten them for FC
        out = self.layer4(out)
        out = self.fc2(out)
        return out

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

model = CNN(num_classes)
model = model.to(device)

def label2onehot(data, class_num):
  datasize = len(data)
  labels = np.zeros((len(data), class_num))
  for i, y in enumerate(data):
      labels[i][y] = 1
  return labels

result_dict = {}
for NOISE_LEVEL in [0.2, 0.5, 0.7, 0.75]:    

    experiments = 10
    experiments_avg_dic = {}

    train_set = WBMs(pkl_dir='../', train=True, NOISE_LEVEL=NOISE_LEVEL, noise_mode=noise_mode)
    test_set = WBMs(pkl_dir='../', train=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    train_loader_track = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False)

    for _ in range(experiments):
	        	    
        Micros = []
        Macros = []
        Accs = []
        Test_losses = []
        Precisions = []
        Recalls = []    

        print('=============== Train for "SELFIE" pattern ===============')
        model, (_, _), model_best = trainSELFIE(model, device, train_loader, test_loader, 
                                  learning_rate, weight_decay, epochs, restart, NOISE_LEVEL, batch_size, test_batch_size, log_interval,
                                  milestones, history_length, reg_term=reg_term)
        model_SELFIE_best = CNN(num_classes).to(device)
        model_SELFIE_best.load_state_dict(torch.load(model_best))
        model_SELFIE_best.eval()

        print('=============== Test for "SELFIE" pattern ===============')
        correct=0
        test_loss=0.0
        MacroF1=0.0
        MicroF1=0.0
        precision = 0.0
        recall = 0.0
        with torch.no_grad():
            for Id, x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                
                raw_output = model_SELFIE_best(x)
                output = torch.log_softmax(raw_output, 1)
                test_loss += F.nll_loss(output, y, reduction='sum')
                
                
                # y_pred_max = y_pred > 0.5
                softmax_output = torch.softmax(raw_output, 1)
                pred = softmax_output.argmax(dim=1, keepdim=True)
                correct += pred.eq(y.view_as(pred)).sum().item()
                y_pred = (softmax_output > 0.5)

                y = label2onehot(y, num_classes)
                MicroF1 += f1_score(y, y_pred.cpu(), average='micro')
                MacroF1 += f1_score(y, y_pred.cpu(), average='macro')
                precision += precision_score(y, y_pred.cpu(), average='macro')
                recall += recall_score(y, y_pred.cpu(), average='macro')

            test_loss /= len(test_loader.dataset)
            precision /= len(test_loader)
            recall /= len(test_loader)
            MicroF1 /= len(test_loader)
            MacroF1 /= len(test_loader)
            # correct /= len(test_loader.dataset)
            Acc = 100*correct/len(test_loader.dataset)
            print('┌ MicroF1: {:.5f}\n│ MacroF1: {:.5f}\n│ Precision: {:.5f}\n│ Recall: {:.5f}\n│ Acc: {}/{} ({:.5f})\n└ Test_loss : {:.5f}'.format(
                MicroF1, MacroF1, precision, recall, correct, len(test_loader.dataset), Acc, test_loss))
            
        Micros.append(MicroF1)
        Macros.append(MacroF1)
        Accs.append(Acc)
        Test_losses.append(test_loss.item())
        Precisions.append(precision)
        Recalls.append(recall)

    experiments_avg_dic['Micro']     = str(np.mean(np.array(Micros), axis=0))+'__'+str(np.std(np.array(Micros), axis=0))
    experiments_avg_dic['Macro']     = str(np.mean(np.array(Macros), axis=0))+'__'+str(np.std(np.array(Macros), axis=0))
    experiments_avg_dic['Acc']       = str(np.mean(np.array(Accs), axis=0))+'__'+str(np.std(np.array(Accs), axis=0))
    experiments_avg_dic['Test_loss'] = str(np.mean(np.array(Test_losses), axis=0))+'__'+str(np.std(np.array(Test_losses), axis=0))
    experiments_avg_dic['Precision'] = str(np.mean(np.array(Precisions), axis=0))+'__'+str(np.std(np.array(Precisions), axis=0))
    experiments_avg_dic['Recall']    = str(np.mean(np.array(Recalls), axis=0))+'__'+str(np.std(np.array(Recalls), axis=0))

    result_dict[NOISE_LEVEL] = experiments_avg_dic

import pandas as pd
result_df = pd.DataFrame.from_dict(result_dict)

from IPython.display import display
display(result_df)

result_df.to_csv("SELFIE_symm.csv")