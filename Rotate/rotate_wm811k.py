import pandas as pd
import numpy as np

import pickle as pkl

for var in ['x_train', 'x_test', 'y_train', 'y_test']:
    with open('../{}.pkl'.format(var), 'rb') as f:
        exec("{} = pkl.load(f)".format(var))

import torch
from torch.utils.data import Dataset

class WBMs(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
#         X = torch.from_numpy(self.X[idx])
#         Y = torch.from_numpy(self.Y[idx])
#         return X, Y
        return idx, self.X[idx], self.Y[idx]

#train_set = WBMs(np.array([x for x in x_train]), np.array(y_train, dtype=np.int16))
#test_set = WBMs(np.array([x for x in x_test]), np.array(y_test, dtype=np.int16))

import os
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

#from util import * 
#from train import *

from sklearn.metrics import f1_score, accuracy_score, average_precision_score

batch_size = 256
test_batch_size = 100

log_interval = 40
learning_rate = 1e-3
weight_decay = 1e-4
reg_term = 0
epochs = 500
milestones = [120, 200, 270, 350, 420]

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

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
#torch.manual_seed(2)  # CPU seed
#if device == "cuda":
#    torch.cuda.manual_seed_all(2)  # GPU seed

### 0. Pre-training : Rotate
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

class Rotate_dataset(Dataset):
    def __init__(self, X, Y, train=False, train_size=0.8, val=False):
        self.X = X
        self.Y = Y
        self.train=train
        self.val=val
        self.train_size = train_size
        
        #X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, train_size=self.train_size, random_state=42)
        #if self.train:
        #    self.X = X_train
        #    self.Y = y_train
        #elif self.val: #
        #    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=self.train_size, random_state=42)
        #    self.X = X_val
        #    self.Y = y_val
        #else:
        #    self.X = X_test
        #    self.Y = y_test
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = self.X[idx].astype(np.float32) # (26, 26, 3)
        X = np.transpose(X, (2, 0, 1)) # (3, 26, 26)
        X_PIL = transforms.ToPILImage()(torch.from_numpy(X))
        rotated_imgs = [
            X_PIL,
            TF.rotate(X_PIL, 90),
            TF.rotate(X_PIL, 180),
            TF.rotate(X_PIL, 270),
        ]
        labels = torch.LongTensor([0, 1, 2, 3])

        index = random.randrange(0, 4)
        X_PIL = rotated_imgs[index]
        Y = labels[index]
        X = transforms.ToTensor()(X_PIL)
        
        return X, Y

Rotate_train_set = Rotate_dataset(X=np.array([x for x in x_train]), Y=y_train, train=True)
Rotate_test_set = Rotate_dataset(X=np.array([x for x in x_test]), Y=y_test)
del x_train, y_train, x_test, y_test

Rotate_train_loader = DataLoader(Rotate_train_set, batch_size=batch_size, shuffle=True)
Rotate_test_lodaer = DataLoader(Rotate_test_set, batch_size=test_batch_size, shuffle=False)

Rotate_model = CNN(4).to(device)

optimizer = optim.Adam(Rotate_model.parameters(), weight_decay=weight_decay, lr=learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

def test_cleaning(model, device, test_loader, test_batch_size, log_interval):
    model.eval()
    loss_per_batch = []
    acc_val_per_batch =[]
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            loss_per_batch.append(F.nll_loss(output, target).item())
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc_val_per_batch.append(100. * correct / ((batch_idx+1)*test_batch_size))

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    loss_per_epoch = [np.average(loss_per_batch)]
    #acc_val_per_epoch = [np.average(acc_val_per_batch)]
    acc_val_per_epoch = [np.array(100. * correct / len(test_loader.dataset))]

    return loss_per_epoch, acc_val_per_epoch

Rotate_model.train()
cont=0
for epoch in range(1,epochs+1):
  scheduler.step()

  loss_per_batch = []
  acc_train_per_batch = []
  correct = 0
  
  for batch_idx, (data, target) in enumerate(Rotate_train_loader):
      data, target = data.to(device), target.to(device)
      optimizer.zero_grad()

      output = Rotate_model(data)
      output = F.log_softmax(output)
      
      loss = F.nll_loss(output, target)
      loss.backward()
      optimizer.step()
      
      loss_per_batch.append(loss.item())

      # save accuracy:
      pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()
      acc_train_per_batch.append(100. * correct / ((batch_idx+1)*batch_size))

      if batch_idx % log_interval == 0:
          print('Epoch: {:>3}/{:>3} | [{:>5}/{:>5} ({:>3.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
              epoch, epochs, batch_idx * len(data), len(Rotate_train_loader.dataset),
                      100. * batch_idx / len(Rotate_train_loader), loss.item(),
                      100. * correct / ((batch_idx + 1) * batch_size),
              optimizer.param_groups[0]['lr']))

  loss_per_epoch = [np.average(loss_per_batch)]
  acc_train_per_epoch = [np.average(acc_train_per_batch)]
  
  # test
  loss_per_epoch_test, acc_val_per_epoch_i = test_cleaning(Rotate_model, device, Rotate_test_lodaer, test_batch_size, log_interval)

  if epoch == 1:
    best_acc_val = acc_val_per_epoch_i[-1]
    best_loss_val = loss_per_epoch_test[-1]
    snapBest = 'Rotate_best_epoch_%d_valLoss_%.5f_valAcc_%.5f_bestAccVal_%.5f' % (
        epoch, loss_per_epoch_test[-1], acc_val_per_epoch_i[-1], best_acc_val)
    directory = './Rotate'
    os.makedirs(directory, exist_ok=True)
    torch.save(Rotate_model.state_dict(), os.path.join(directory, snapBest + '.pth'))
    print('└ save the best model in {}/{}\n'.format(directory, snapBest+'.pth')) 
    #torch.save(optimizer.state_dict(), os.path.join(directory, 'opt_' + snapBest + '.pth'))
    
  else:
      if (acc_val_per_epoch_i[-1] >= best_acc_val) and (loss_per_epoch_test[-1] < best_loss_val):
          best_acc_val = acc_val_per_epoch_i[-1]
          best_loss_val = loss_per_epoch_test[-1]
          try:
              os.remove(os.path.join(directory, snapBest + '.pth'))
          except:
              pass          
          snapBest = 'Rotate_best_epoch_%d_valLoss_%.5f_valAcc_%.5f_bestAccVal_%.5f' % (
              epoch, loss_per_epoch_test[-1], acc_val_per_epoch_i[-1], best_acc_val)
          torch.save(Rotate_model.state_dict(), os.path.join(directory, snapBest + '.pth'))
          print('└ save the best model in {}/{}\n'.format(directory, snapBest+'.pth')) 
          #torch.save(optimizer.state_dict(), os.path.join(directory, 'opt_' + snapBest + '.pth'))

  cont+=1

  if epoch == epochs:
      snapLast = 'Rotate_last_epoch_%d_valLoss_%.5f_valAcc_%.5f_bestValLoss_%.5f' % (
          epoch, loss_per_epoch_test[-1], acc_val_per_epoch_i[-1], best_acc_val)
      #torch.save(model.state_dict(), os.path.join(directory, snapLast + '.pth'))
      #torch.save(optimizer.state_dict(), os.path.join(directory, 'opt_' + snapLast + '.pth'))