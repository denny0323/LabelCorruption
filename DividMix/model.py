#!/usr/bin/env python
# coding: utf-8

# In[7]:

import random
import matplotlib
import numpy as np
import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F


from torchvision import transforms
from torch.utils.data import Dataset


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
    
    
    
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv = nn.Sequential(
            # 1*56*53   2
            nn.Conv2d(1, 64, 5, padding=2),nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 5, padding=2),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, padding=(0,1), dilation=(1,2)),
            # 64*28*27   2
            nn.Conv2d(64, 128, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, padding=(0,1), dilation=(1,2)),
            # 128*14*14   3
            nn.Conv2d(128, 256, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 256*7*7   3
            nn.Conv2d(256, 512, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, padding=1, dilation=2),
            
            #512 4*4   3
            nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2)  # out: 512*2*2
        )
        #self.avg_pool = nn.AvgPool2d(7)
        
        self.fc1 = nn.Linear(512*2*2, 1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,4)
        

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1) # flatten
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        return x
    
    
    
class SmallVGG(nn.Module):
    def __init__(self):
        super(SmallVGG, self).__init__()
        self.conv = nn.Sequential(
            # (?, 1, 56, 53)   1
            nn.Conv2d(1, 32, 5, padding=2),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, padding=(0,1), dilation=(1,2)),
            
            # (?, 32, 28, 27)   2
            nn.Conv2d(32, 64, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, padding=(0,1), dilation=(1,2)),
            
            # (?, 64, 14, 14)   2
            nn.Conv2d(64, 128, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, padding=(0,1), dilation=(1,2)),
            
            # (?, 128, 7, 7)   3
            nn.Conv2d(128, 512, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, padding=1, dilation=2) # out (128, 4, 4)
        )
        #self.avg_pool = nn.AvgPool2d(7)
        
        self.fc1 = nn.Linear(512*4*4, 4096)
        self.fc2 = nn.Linear(4096, 512)
        self.fc3 = nn.Linear(512,4)
        

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1) # flatten
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        return x