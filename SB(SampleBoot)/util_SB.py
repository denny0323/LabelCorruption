#!/usr/bin/env python
# coding: utf-8

# In[7]:

import random
import matplotlib
import numpy as np
import pickle as pkl

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

class WBMs(Dataset):
    def __init__(self, pkl_dir='../', train=True, NOISE_LEVEL=False, noise_mode='symm', contain=False):

        self.pkl_dir = pkl_dir
        self.train = train  
        self.NOISE_LEVEL = NOISE_LEVEL
        self.noise_mode = noise_mode
        self.contain = contain # when relabelling, whether containing its original label, --default: False

        self.transition = {0:6, 1:2, 2:4, 3:1, 4:3, 5:0, 6:5}

        if self.train:
            with open(self.pkl_dir+'x_train.pkl', 'rb') as fx:
                self.X = pkl.load(fx)
                self.X = torch.from_numpy(self.X)
                self.X = self.X.permute(0, 3, 1, 2).float()

            with open(self.pkl_dir+'y_train.pkl', 'rb') as fy:
                self.Y = pkl.load(fy)

        else:
            with open(self.pkl_dir + 'x_test.pkl', 'rb') as fx:
                self.X = pkl.load(fx)
                self.X = torch.from_numpy(self.X)
                self.X = self.X.permute(0, 3, 1, 2).float()

            with open(self.pkl_dir + 'y_test.pkl', 'rb') as fy:
                self.Y = pkl.load(fy)

        
        if self.NOISE_LEVEL:
            indices = list(range(len(self.X)))
            random.shuffle(indices)

            num_noise = int(self.NOISE_LEVEL * len(self.Y))
            noise_idx = indices[:num_noise]

            y_noise = []
            for i in range(len(indices)):
                label_list = list(range(7))
                if i in noise_idx:
                    if noise_mode == 'symm':
                        if not contain:  # 자기자신 포함해서 바꿀건지
                            label_list.pop(self.Y[i])
                        noiselabel = random.choice(label_list)
                        y_noise.append(noiselabel)

                    elif noise_mode == 'asymm':
                        noiselabel = self.transition[self.Y[i]]
                        y_noise.append(noiselabel)

                else:
                    y_noise.append(self.Y[i])  # true label도 포함

            # X_train = np.array([x for x in X_train])
            self.Y = np.asarray(y_noise)

            
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return idx, self.X[idx], self.Y[idx]
        
    
class Single_dataset(Dataset):
    def __init__(self, dataset):
        self.single_indices = [i for i, (x, y) in enumerate(dataset) if np.sum(y, 0) == 1]
        self.X, self.Y = dataset[self.single_indices]
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    

class Rotate_dataset(Dataset):
    def __init__(self, dataset, train=False, train_size=0.8, val=False):
        self.train = train
        self.val = val
        self.X = dataset.X
        self.Y = dataset.Y
        self.train_size = train_size
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, train_size=self.train_size, random_state=42)
        if self.train:
            self.X = X_train
            self.Y = y_train
        elif self.val: #
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=self.train_size, random_state=42)
            self.X = X_val
            self.Y = y_val
        else:
            self.X = X_test
            self.Y = y_test
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = self.X[idx]
        X_PIL = transforms.ToPILImage()(X.squeeze_(0))
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
    
    
    
class MiniBatch(object):
    def __init__(self):
        self.ids = []
        self.images = []
        self.labels = []

    def append(self, id, image, label):
        self.ids.append(id)
        self.images.append(image)
        self.labels.append(label)

    def get_size(self):
        return len(self.ids)
    
    
    
class History_Bank(object):
    def __init__(self, size_of_data, history_length, num_of_classes, threshold=0.05, loaded_data=None):
        self.size_of_data = size_of_data
        self.history_length = history_length
        self.num_of_classes = num_of_classes
        self.threshold = threshold  
        
        self.learned_labels = {}
        for i in range(self.size_of_data):
            self.learned_labels[i] = np.zeros(self.history_length, dtype=int)
        
        self.corrected_labels = {}
        for i in range(size_of_data):
            self.corrected_labels[i] = -1
            
        self.max_uncertainty = -np.log(1.0/float(self.num_of_classes))
        
        self.update_counter = np.zeros(self.size_of_data, dtype=int)
        
        # For Logging
        self.loaded_data = None
        if loaded_data is not None:
            self.loaded_data = loaded_data
        
    def write_label_history(self, Ids, y, init=False):
        for index, Id in enumerate(Ids):
            id = int(Id)
            cur_index = self.update_counter[id] % self.history_length
            self.learned_labels[id][cur_index] = y[index]
            if not init:
                self.update_counter[id] += 1
                
    def get_refurbishable_samples(self, ids, images):
        corrected_batch = MiniBatch()

        # check predictive uncertainty
        for i in range(len(ids)):
            id = ids[i]
            image = images[i]
            
            predictions = self.learned_labels[id]
            
            p_dict = np.zeros(self.num_of_classes, dtype=float)
            for _, value in enumerate(predictions):
                p_dict[value] += float(1) / float(10)

            # compute predictive uncertainty
            negative_entropy = 0.0
            for i in range(len(p_dict)):
                if p_dict[i] == 0:
                    negative_entropy += 0.0
                else:
                    negative_entropy += p_dict[i] * np.log(p_dict[i])
            uncertainty = - negative_entropy / self.max_uncertainty

            ############### correspond to the lines 12--19 of the paper ################
            # check refurbishable condition
            if uncertainty >= (1-self.threshold):
                self.corrected_labels[id] = np.argmax(p_dict)
                corrected_batch.append(id, image, self.corrected_labels[id])

                # For logging ###########################################################
                #if self.loaded_data is not None:
                #    self.loaded_data[id].corrected = True
                #    self.loaded_data[id].last_corrected_label = self.corrected_labels[id]
                #########################################################################

            # reuse previously classified refurbishalbe samples
            # As we tested, this part degraded the performance marginally around 0.3%p
            # because uncertainty of the sample may afftect the performance
            elif self.corrected_labels[id] != -1:
                corrected_batch.append(id, image, self.corrected_labels[id])

        return corrected_batch    
    