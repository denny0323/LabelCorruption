from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
import pickle as pkl
import torch.nn.functional as F

from torchnet.meter import AUCMeter

def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


class WBMs(Dataset):
    def __init__(self, r, noise_mode, mode, transform, contain=False, pkl_dir='../', pred=[], probability=[], log=''):
        self.r = r # noise ratio
        self.noise_mode = noise_mode
        self.mode = mode
        self.contain = contain
        self.pkl_dir = pkl_dir
        self.transform = transform
        self.transition = {0:6, 1:2, 2:4, 3:1, 4:3, 5:0, 6:5}

        if self.mode=='test':
            with open(self.pkl_dir+'x_test.pkl', 'rb') as fx:
                self.X = pkl.load(fx)
                self.X = torch.from_numpy(self.X)
                self.X = self.X.permute(0, 3, 1, 2).float()

            with open(self.pkl_dir+'y_test.pkl', 'rb') as fy:
                self.Y = pkl.load(fy)
                
        else:
            #if self.mode=='train':
            with open(self.pkl_dir+'x_train.pkl', 'rb') as fx:
                self.X = pkl.load(fx)
                self.X = torch.from_numpy(self.X)
                self.X = self.X.permute(0, 3, 1, 2).float()

            with open(self.pkl_dir+'y_train.pkl', 'rb') as fy:
                self.Y = pkl.load(fy)

            if self.r:

                indices = list(range(len(self.X)))
                random.shuffle(indices)

                num_noise = int(self.r * len(self.Y))
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

                #X_train = np.array([x for x in X_train])
                self.Y = np.asarray(y_noise)
                

            if self.mode == 'all':
                pass

            else:                   
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]   

                    clean = (y_noise==self.Y)

                    auc_meter = AUCMeter()
                    auc_meter.reset()
                    auc_meter.add(probability, clean)        
                    auc,_,_ = auc_meter.value()               
                    log.write('Numer of labeled samples:%d   AUC:%.3f\n'%(pred.sum(),auc))
                    log.flush()      

                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]                                               
                
                self.X = self.X[pred_idx]
                self.Y = [y_noise[i] for i in pred_idx]
                print("%s data has a size of %d"%(self.mode, len(self.Y)))            


    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target, prob = self.X[index], self.Y[index], self.probability[index]
            #img = Image.fromarray(img)
            img = transforms.ToPILImage()(img)
            img1 = self.transform(img)
            img2 = self.transform(img)
            return transforms.ToTensor()(img1),  transforms.ToTensor()(img2), target, prob  
        
        elif self.mode=='unlabeled':
            img = self.X[index]
            img = transforms.ToPILImage()(img)
            img1 = self.transform(img)
            img2 = self.transform(img)
            #img = Image.fromarray(img)
            return transforms.ToTensor()(img1), transforms.ToTensor()(img2)
        
        elif self.mode=='all':
            img, target = self.X[index], self.Y[index]
            #img = Image.fromarray(img)
            return img, target, index    
        
        elif self.mode=='test':
            img, target = self.X[index], self.Y[index]
            #img = Image.fromarray(img)  
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.X)
        else:
            return len(self.X)         
    
    
class WBMs_dataloader():  
    def __init__(self, r, noise_mode, batch_size, pkl_dir, log, num_workers):

        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.pkl_dir = pkl_dir
        self.log = log
        self.num_workers = num_workers
        self.transform_train = transforms.Compose([
                    transforms.RandomCrop(26, padding=4),
                    transforms.RandomHorizontalFlip(),
                ]) 
        self.transform_test = transforms.Compose([
                ])      
    
    def run(self,mode,pred=[],prob=[]):
        if mode=='warmup':
            all_dataset = WBMs(r=self.r, noise_mode=self.noise_mode, pkl_dir=self.pkl_dir, mode="all", transform=self.transform_train)
            
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader
                                     
        elif mode=='train':
            labeled_dataset = WBMs(r=self.r, noise_mode=self.noise_mode, pkl_dir=self.pkl_dir, mode="labeled", transform=self.transform_train, pred=pred, log=self.log, probability=prob)
            
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            
            unlabeled_dataset =WBMs(r=self.r, noise_mode=self.noise_mode, pkl_dir=self.pkl_dir, mode="unlabeled", transform=self.transform_train, pred=pred, probability=prob)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)     
            return labeled_trainloader, unlabeled_trainloader
        
        elif mode=='test':
            test_dataset =WBMs(r=self.r, noise_mode=self.noise_mode, pkl_dir=self.pkl_dir, mode="test", transform=self.transform_test)
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = WBMs(r=0.000, noise_mode=self.noise_mode, pkl_dir=self.pkl_dir, mode="all", transform=self.transform_test)
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader        