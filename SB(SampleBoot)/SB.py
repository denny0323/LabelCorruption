import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.optim as optim
import gc
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib
font = {'size': 14}
matplotlib.rc('font', **font)

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import f1_score, accuracy_score, average_precision_score, precision_score, recall_score
from train_single import *

torch.cuda.is_available()

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
# torch.manual_seed(2)  # CPU seed
# if device == "cuda":
#     torch.cuda.manual_seed_all(2)  # GPU seed

import pickle as pkl

#for var in ['x_train', 'x_test', 'y_train', 'y_test']:
#  with open('../{}.pkl'.format(var), 'rb') as f:
#     exec("{} = pkl.load(f)".format(var))

from util_SB import *

import random
transition = list(range(7))
random.shuffle(transition)
transition = {i:v for i, v in enumerate(transition)}

def label2onehot(data, class_num=7):
  datasize = len(data)
  labels = np.zeros((len(data), class_num))
  for i, y in enumerate(data):
      labels[i][y] = 1
  return labels

import torch
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

learning_rate = 1e-5
weight_decay = 1e-3
reg_term = 0
restart = 1
epochs = 60
warmup = 20
max_iters = 15
sample_ratio = 0.9
milestones = [epochs, epochs]

from util_SB import *
from collections import defaultdict

experiment_results = {}
for NOISE_LEVEL in [0.2, 0.5, 0.7, 0.75]:
    
    Hard_vals = defaultdict(list)
    Soft_vals = defaultdict(list)
    
    batch_size = 128
    test_batch_size = 64

    noise_mode='symm'
    contain = False

    experiments = 10
    for _ in range(experiments):

        train_set = WBMs(train=True, NOISE_LEVEL=NOISE_LEVEL, noise_mode=noise_mode)
        train_set_track = WBMs(train=True, NOISE_LEVEL=0.000)
        test_set = WBMs(train=False)


        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        train_loader_track = DataLoader(train_set_track, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False)
        del train_set, test_set

        log_interval = int(len(train_loader)/2)

        ### 1. Load pre-trained model
        bestRotmodel = '../Rotate/Rotate/Rotate_best_epoch_99_valLoss_0.23348_valAcc_93.22971_bestAccVal_93.22971.pth'
        pretrained_dict = torch.load(bestRotmodel, map_location="cuda:0")
        filtered_dict = {k: v for k, v in pretrained_dict.items() if k not in ['fc2.weight', 'fc2.bias']}

        model_hard = CNN(7)
        # model = torch.nn.DataParallel(model)
        model_hard = model_hard.to(device)
        model_hard_dict = model_hard.state_dict()
        model_hard_dict.update(filtered_dict)
        model_hard.load_state_dict(model_hard_dict)

        ### Hard
        print('=============== Train for WM-811k (26 * 26) sizes ===============')
        model_hard, (losses, accs), bestmodel_hard = sample_boot(model_hard, device, train_loader, train_loader_track, test_loader, learning_rate, weight_decay, 
                                                      epochs, restart, 'BMM', max_iters, NOISE_LEVEL, sample_ratio, batch_size, test_batch_size, log_interval, milestones, reg_term=reg_term, TYPE='Hard')
        del model_hard, (losses, accs)
        ### Test
        model_hard_best = CNN(7).to(device)
        model_hard_best.load_state_dict(torch.load(bestmodel_hard))
        #print(bestmodel_hard)

        model_hard_best.eval()

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
                
                raw_output = model_hard_best(x)
                output = torch.log_softmax(raw_output, 1)
                test_loss += F.nll_loss(output, y.long(), reduction='sum')
                
                
                # y_pred_max = y_pred > 0.5
                softmax_output = torch.softmax(raw_output, 1)
                pred = softmax_output.argmax(dim=1, keepdim=True)
                correct += pred.eq(y.view_as(pred)).sum().item()
                y_pred = (softmax_output > 0.5)
                
                y = label2onehot(y, 7)
                MicroF1 += f1_score(y, y_pred.cpu(), average='micro')
                MacroF1 += f1_score(y, y_pred.cpu(), average='macro')
                precision += precision_score(y, y_pred.cpu(), average='macro')
                recall += recall_score(y, y_pred.cpu(), average='macro')

            test_loss /= len(test_loader.dataset)
            precision /= len(test_loader)
            recall /= len(test_loader)
            MicroF1 /= len(test_loader)
            MacroF1 /= len(test_loader)
            correct /= len(test_loader.dataset)
            Acc = 100*correct/len(test_loader.dataset)
            print('┌ MicroF1: {:.5f}\n│ MacroF1: {:.5f}\n│ Precision: {:.5f}\n│ Recall: {:.5f}\n│ Acc: {}/{} ({:.5f})\n└ Test_loss : {:.5f}'.format(
                MicroF1, MacroF1, precision, recall, correct, len(test_loader.dataset), Acc, test_loss))
            
        Hard_vals['Micro'].append(MicroF1)
        Hard_vals['Macro'].append(MacroF1)
        Hard_vals['Acc'].append(Acc)
        Hard_vals['Test_loss'].append(test_loss.cpu().numpy())
        Hard_vals['Precision'].append(precision)
        Hard_vals['Recall'].append(recall)
        del model_hard_best, MicroF1, MacroF1, Acc, test_loss, precision, recall

        torch.cuda.empty_cache()


        model_soft = CNN(7)
        # model = torch.nn.DataParallel(model)
        model_soft = model_soft.to(device)
        model_soft_dict = model_soft.state_dict()
        model_soft_dict.update(filtered_dict)
        model_soft.load_state_dict(model_soft_dict)

        ### Soft
        print('=============== Train for WM-811k (26 * 26) sizes ===============')
        model_soft, (losses, accs), bestmodel_soft = sample_boot(model_soft, device, train_loader, train_loader_track, test_loader, learning_rate, weight_decay, 
                                                      epochs, restart, 'BMM', max_iters, NOISE_LEVEL, sample_ratio, batch_size, test_batch_size, log_interval, milestones, reg_term=reg_term, TYPE='Soft')
        ### Test
        model_soft_best = CNN(7).to(device)
        model_soft_best.load_state_dict(torch.load(bestmodel_soft))
        #print(bestmodel_soft)

        model_soft_best.eval()

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
                
                raw_output = model_soft_best(x)
                output = torch.log_softmax(raw_output, 1)
                test_loss += F.nll_loss(output, y.long(), reduction='sum')
                
                
                # y_pred_max = y_pred > 0.5
                softmax_output = torch.softmax(raw_output, 1)
                pred = softmax_output.argmax(dim=1, keepdim=True)
                correct += pred.eq(y.view_as(pred)).sum().item()
                y_pred = (softmax_output > 0.5)
                
                y = label2onehot(y, 7)
                MicroF1 += f1_score(y, y_pred.cpu(), average='micro')
                MacroF1 += f1_score(y, y_pred.cpu(), average='macro')
                precision += precision_score(y, y_pred.cpu(), average='macro')
                recall += recall_score(y, y_pred.cpu(), average='macro')

            test_loss /= len(test_loader.dataset)
            precision /= len(test_loader)
            recall /= len(test_loader)
            MicroF1 /= len(test_loader)
            MacroF1 /= len(test_loader)
            correct /= len(test_loader.dataset)
            Acc = 100*correct/len(test_loader.dataset)
            print('┌ MicroF1: {:.5f}\n│ MacroF1: {:.5f}\n│ Precision: {:.5f}\n│ Recall: {:.5f}\n│ Acc: {}/{} ({:.5f})\n└ Test_loss : {:.5f}'.format(
                MicroF1, MacroF1, precision, recall, correct, len(test_loader.dataset), Acc, test_loss))
            
        Soft_vals['Micro'].append(MicroF1)
        Soft_vals['Macro'].append(MacroF1)
        Soft_vals['Acc'].append(Acc)
        Soft_vals['Test_loss'].append(test_loss.float().cpu().numpy())
        Soft_vals['Precision'].append(precision)
        Soft_vals['Recall'].append(recall)

        del model_soft_best, MicroF1, MacroF1, Acc, test_loss, precision, recall
   

    import numpy as np
    Hard_vals_Micro = Hard_vals['Micro']
    Hard_vals['Micro']     = np.mean(np.array(Hard_vals_Micro), axis=0)
    Hard_vals['Micro_std'] = np.std(np.array(Hard_vals_Micro), axis=0)

    Hard_vals_Macro = Hard_vals['Macro']
    Hard_vals['Macro']     = np.mean(np.array(Hard_vals_Macro), axis=0)
    Hard_vals['Macro_std'] = np.std(np.array(Hard_vals_Macro), axis=0)

    Hard_vals_Acc = Hard_vals['Acc']
    Hard_vals['Acc']       = np.mean(np.array(Hard_vals_Acc), axis=0)
    Hard_vals['Acc_std']   = np.std(np.array(Hard_vals_Acc), axis=0)

    Hard_vals_Test_loss = Hard_vals['Test_loss']
    Hard_vals['Test_loss'] = np.mean(np.array(Hard_vals_Test_loss), axis=0)
    Hard_vals['loss_std']  = np.std(np.array(Hard_vals_Test_loss), axis=0)

    Hard_vals_Precision = Hard_vals['Precision']
    Hard_vals['Precision'] = np.mean(np.array(Hard_vals_Precision), axis=0)
    Hard_vals['Pre_std']   = np.std(np.array(Hard_vals_Precision), axis=0)

    Hard_vals_Recall = Hard_vals['Recall']
    Hard_vals['Recall']    = np.mean(np.array(Hard_vals_Recall), axis=0)
    Hard_vals['Rec_std']   = np.std(np.array(Hard_vals_Recall), axis=0)

    
    Soft_vals_Micro = Soft_vals['Micro']
    Soft_vals['Micro']     = np.mean(np.array(Soft_vals_Micro), axis=0)
    Soft_vals['Micro_std'] = np.std(np.array(Soft_vals_Micro), axis=0)

    Soft_vals_Macro = Soft_vals['Macro']
    Soft_vals['Macro']     = np.mean(np.array(Soft_vals_Macro), axis=0)
    Soft_vals['Macro_std'] = np.std(np.array(Soft_vals_Macro), axis=0)

    Soft_vals_Acc = Soft_vals['Acc']
    Soft_vals['Acc']       = np.mean(np.array(Soft_vals_Acc), axis=0)
    Soft_vals['Acc_std']   = np.std(np.array(Soft_vals_Acc), axis=0)

    Soft_vals_Test_loss = Soft_vals['Test_loss']
    Soft_vals['Test_loss'] = np.mean(np.array(Soft_vals_Test_loss), axis=0)
    Soft_vals['loss_std']  = np.std(np.array(Soft_vals_Test_loss), axis=0)

    Soft_vals_Precision = Soft_vals['Precision']
    Soft_vals['Precision'] = np.mean(np.array(Soft_vals_Precision), axis=0)
    Soft_vals['Pre_std']   = np.std(np.array(Soft_vals_Precision), axis=0)

    Soft_vals_Recall = Soft_vals['Recall']
    Soft_vals['Recall']    = np.mean(np.array(Soft_vals_Recall), axis=0)
    Soft_vals['Rec_std']   = np.std(np.array(Soft_vals_Recall), axis=0)

    experiment_results[NOISE_LEVEL] = {'Hard_vals': Hard_vals, 'Soft_vals': Soft_vals}

print(experiment_results)

import pandas as pd
result_df = pd.DataFrame.from_dict({(i,j):experiment_results[i][j]
                        for i in experiment_results.keys()
                        for j in experiment_results[i].keys()},
                        )

from IPython.display import display
display(result_df)

result_df.to_csv("SB_symm.csv")
