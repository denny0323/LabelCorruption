#!/usr/bin/env python
# coding: utf-8

# In[7]:

import scipy.stats as stats
import sys, os, math
import random
import matplotlib
import numpy as np
import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, Subset, WeightedRandomSampler
#from pytorchtools import EarlyStopping
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture as GMM

# from util import *
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
torch.manual_seed(2)  # CPU seed
if device == "cuda":
    torch.cuda.manual_seed_all(2)  # GPU seed
        
def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)

def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar)**2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) /x_bar
    return alpha, beta

def Normalized(loss_tr):
    max_perc = np.percentile(loss_tr, 95)
    min_perc = np.percentile(loss_tr, 5)
    loss_tr = loss_tr[(loss_tr<=max_perc) & (loss_tr>=min_perc)]

    maxLoss = torch.FloatTensor([max_perc]).to(device)
    minLoss = torch.FloatTensor([min_perc]).to(device) + 10e-6
    
    loss_tr = (loss_tr - minLoss.data.cpu().numpy()) / (maxLoss.data.cpu().numpy() - minLoss.data.cpu().numpy() + 1e-6)
    loss_tr[loss_tr>=1] = 1-10e-4
    loss_tr[loss_tr<= 0] = 10e-4

    return loss_tr, maxLoss, minLoss


def Weightedsampler(sample_size, W):
    weight = torch.abs(W-0.5)
    #weight /= torch.max(weight.clone()) # max값으로 normalizing...?
    #weight /= torch.sum(weight.clone()) # 전체 값으로 normalizing ?
    weight[weight==0.0] = 1e-10
    sampler = WeightedRandomSampler(weight, sample_size, replacement=False)
    indices = []
    for idx in sampler:
        indices.append(idx)
    return indices

def zero_division(n, ids):
    d = len(ids)
    return n / d if d else np.nan

def compute_Recall(Ids, dic, label='Clean'):
    if len(Ids) != 1:
        Ids=Ids.squeeze()
        
    n = 0
    if Ids.size()[0] != 0:
        for Id in Ids:
            Id = int(Id)
            if dic[Id][1] == label:
                n+=1
                
    return zero_division(n, Ids)

class BetaMixture1D(object):
    def __init__(self, max_iters=10,
                 alphas_init=[1, 2],
                 betas_init=[2, 1],
                 weights_init=[0.5, 0.5]):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r =  np.array([self.weighted_likelihood(x, i) for i in range(2)])
        # there are ~200 samples below that value
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, x):
        x = np.copy(x)

        # EM on beta distributions unsable with x == 0 or 1
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        for i in range(self.max_iters):

            # E-step
            r = self.responsibilities(x)

            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()

        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def create_lookup(self, y):
        x_l = np.linspace(0+self.eps_nan, 1-self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t
        self.lookup_loss = x_l # I do not use this one at the end

    def look_lookup(self, x, loss_max, loss_min):
        x_i = x.clone().cpu().numpy()
        x_i = np.array((self.lookup_resolution * x_i).astype(int))
        x_i[x_i < 0] = 0
        x_i[x_i == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x_i]

    def plot(self):
        x = np.linspace(0, 1, 100)
        #plt.plot(x, self.weighted_likelihood(x, 0), label='negative')
        #plt.plot(x, self.weighted_likelihood(x, 1), label='positive')
        plt.plot(x, self.probability(x), 'k--', lw=2, label='Beta mixture')
        plt.legend(loc='upper right')
        
    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)
    
class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))
    
def reg_loss_class(mean_tab,num_classes=7):
    loss = 0
    for items in mean_tab:
        loss += (1./num_classes)*torch.log((1./num_classes)/items)
    return loss
    

def train_CrossEntropy(model, device, train_loader, optimizer, epoch, epochs, batch_size, log_interval):
    model.train()
    loss_per_batch = []
    acc_train_per_batch = []
    correct = 0
    
    for batch_idx, (Id, data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
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
                epoch, epochs, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                       100. * correct / ((batch_idx + 1) * batch_size),
                optimizer.param_groups[0]['lr']))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [np.average(acc_train_per_batch)]
    return (loss_per_epoch, acc_train_per_epoch)    
    
    
def train_pseudo(model, device, train_loader, optimizer, epoch, epochs, batch_size, log_interval):
    
    model.train()
    loss_per_batch = []
    acc_train_per_batch = []
    correct = 0
    
    trackdic = train_loader.dataset.sample_trackdic
    
    for batch_idx, (Id, data, target) in enumerate(train_loader):
        Id, data, target = Id.to(device), data.to(device), target.long().to(device)
        optimizer.zero_grad()

        output = model(data)
        output = F.log_softmax(output)
        new_target = target.clone()
        target_flipped = 1-new_target
        
        loss = F.nll_loss(output, target, reduction='none')
        
        for i in range(loss.size()[0]):
            id = Id[i].item()
            output_ = output[i].unsqueeze(0)
            target_flipped_ = target_flipped[i]
            loss_flip = F.nll_loss(output_, target_flipped_.unsqueeze(0))
            
            if loss[i] > loss_flip:
                trackdic[id][0] = target_flipped_.item()  # label correction
                if trackdic[id][1] == 'Noise':
                    trackdic[id][3] = True               # label correction
                new_target[i] = target_flipped_                  # loss correction by correcting target
        
        loss = F.nll_loss(output, new_target)      
        loss.backward()
        optimizer.step()

        loss_per_batch.append(loss.item())
        
        # save accuracy:
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        acc_train_per_batch.append(100. * correct / ((batch_idx+1)*batch_size))

        if batch_idx % log_interval == 0:
            print('Epoch: {:>3}/{:>3} | [{:>5}/{:>5} ({:>3.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, epochs, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                       100. * correct / ((batch_idx + 1) * batch_size),
                optimizer.param_groups[0]['lr']))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [np.average(acc_train_per_batch)]
    return (loss_per_epoch, acc_train_per_epoch)    


    
def warmup_CE(model, device, train_loader, historybank, optimizer, epoch, epochs, batch_size, log_interval):
    model.train()
    loss_per_batch = []
    acc_train_per_batch = []
    correct = 0
    
    for batch_idx, (Id, data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
          
        optimizer.zero_grad()
        
        output = model(data)
        output = F.log_softmax(output, dim=1)
        
        loss = F.nll_loss(output, target)        
        loss.backward()
        optimizer.step()
        
        loss_per_batch.append(loss.item())

        # save accuracy:
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        historybank.write_label_history(Id, pred)
        correct += pred.eq(target.view_as(pred)).sum().item()
        acc_train_per_batch.append(100. * correct / ((batch_idx+1)*batch_size))

        if batch_idx % log_interval == 0:
            print('Epoch: {:>3}/{:>3} | [{:>5}/{:>5} ({:>3.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, epochs, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                       100. * correct / ((batch_idx + 1) * batch_size),
                optimizer.param_groups[0]['lr']))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [np.average(acc_train_per_batch)]
    return loss_per_epoch, acc_train_per_epoch, historybank
    
    
    
def track_training_loss(model, device, train_loader, epoch, mixture, max_iters, bmm_model1, bmm_model_maxLoss1, bmm_model_minLoss1):
    
    model.eval()

    all_losses = torch.Tensor()
    all_predictions = torch.Tensor()
    all_probs = torch.Tensor()
    all_argmaxXentropy = torch.Tensor()

    for batch_idx, (Id, data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        logit = model(data)
        prediction = F.log_softmax(logit, dim=1)
        
        idx_loss = F.nll_loss(prediction, target, reduction = 'none')
        idx_loss.detach_()
        idx_loss = F.cross_entropy(prediction, target, reduction = 'none')
        idx_loss.detach_()
        
        all_losses = torch.cat((all_losses, idx_loss.cpu()))
        
        probs = prediction.clone()
        probs.detach_()
        all_probs = torch.cat((all_probs, probs.cpu()))
        
        arg_entr = torch.max(prediction, dim=1)[1]
        arg_entr = F.nll_loss(prediction.float(), arg_entr.to(device), reduction='none')
        arg_entr.detach_()
        all_argmaxXentropy = torch.cat((all_argmaxXentropy, arg_entr.cpu()))

    loss_tr = all_losses.data.numpy()

    if mixture=='BMM':
        
        norm_loss_tr, bmm_model_maxLoss, bmm_model_minLoss = Normalized(loss_tr)
        
        bmm_model = BetaMixture1D(max_iters=max_iters)
        bmm_model.fit(norm_loss_tr)
        bmm_model.create_lookup(1)

        # plt.hist(norm_loss_tr, bins=100, color='g', density=1, label='All')
        # bmm_model.plot()
        # plt.show()
        # plt.close()
        
        return all_losses.data.numpy(), \
           all_probs.data.numpy(), \
           all_argmaxXentropy.numpy(), \
           bmm_model, bmm_model_maxLoss, bmm_model_minLoss

    elif mixture=='GMM':
        
        #loss_tr = np.log(loss_tr)
        plt.hist(loss_tr, bins=100, color='g', density=1, label='All')
        
        # GMM plot
        gmm = GMM(n_components=2, max_iter=max_iters)
        gmm.fit(loss_tr.reshape(-1,1))
        x = np.linspace(math.floor(min(loss_tr)), math.ceil(max(loss_tr)), 1000)
        y = np.exp(gmm.score_samples(x.reshape(-1,1)))
        # plt.plot(x, y, 'k--', label='GMM')
        # plt.legend(loc='upper right')
        # plt.show()
        
        prob = gmm.predict_proba(loss_tr.reshape(-1,1)) 
        prob = prob[:,gmm.means_.argmax()]        
        
        return all_losses.data.numpy(), prob, None, gmm, None, None
    
def test_cleaning(model, device, test_loader, test_batch_size, log_interval):
    model.eval()
    loss_per_batch = []
    acc_val_per_batch =[]
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (Id, data, target) in enumerate(test_loader):
            data, target = data.to(device), target.long().to(device)
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


def compute_probabilities_batch(data, target, cnn_model, mixture_model, bmm_model_maxLoss, bmm_model_minLoss, mixture='BMM'):
    
    cnn_model.eval()
    outputs = cnn_model(data)
    outputs = F.log_softmax(outputs, dim=1)
    
    batch_losses = F.nll_loss(outputs.float(), target, reduction = 'none')
    batch_losses.detach_()
    outputs.detach_()
    cnn_model.train()
    
    if mixture=='BMM':
        batch_losses = (batch_losses - bmm_model_minLoss) / (bmm_model_maxLoss - bmm_model_minLoss + 1e-6)
        batch_losses[batch_losses >= 1] = 1-10e-4
        batch_losses[batch_losses <= 0] = 10e-4

        #B = bmm_model.posterior(batch_losses, 1)
        B = mixture_model.look_lookup(batch_losses, bmm_model_maxLoss, bmm_model_minLoss)
        return torch.FloatTensor(B)
    
    elif mixture=='GMM':
        prob = mixture_model.predict_proba(batch_losses.reshape(-1,1)) 
        prob = prob[:,mixture_model.means_.argmax()]
        return torch.FloatTensor(prob)

    
def train_DYN_Bootstrapping(model, device, train_loader, optimizer, epoch, epochs, TYPE, mixture, mixture_model, \
                            bmm_model_maxLoss, bmm_model_minLoss, batch_size, log_interval, reg_term=0):
    model.train()
    loss_per_batch = []
    acc_train_per_batch = []
    correct = 0
    conf_penalty = NegEntropy()
    
    for batch_idx, (Id, data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        output.detach_() 
        optimizer.zero_grad()
        
        output_mean = F.softmax(output, dim=1)
        tab_mean_class = torch.mean(output_mean, -2)
        
        if mixture == 'BMM':
            W = compute_probabilities_batch(data, target, model, mixture_model, bmm_model_maxLoss, bmm_model_minLoss, mixture=mixture)
            W = B.to(device)
            W[W <= 1e-4] = 1e-4
            W[W >= 1 - 1e-4] = 1 - 1e-4 # w
        
        elif mixture == 'GMM':
            W = compute_probabilities_batch(data, target, model, mixture_model, 0, 0, mixture=mixture)
              
        if TYPE == 'Hard':
            output = F.log_softmax(output, dim=1) # h(x)
            z = torch.max(output, dim=1)[1] # z
            losses = (1-W) * F.nll_loss(output, target, reduction='none') + \
                     (  W) * F.nll_loss(output, z, reduction='none')
            
            loss = torch.sum(losses) / len(losses)

        elif TYPE == 'Soft':
            pred = F.log_softmax(output, dim=1) # log(h(x))
            losses = (1-W) * F.nll_loss(pred, target, reduction='none') + \
                     (  W) * -torch.sum(F.softmax(output,dim=1) * pred, dim=1) # F.softmax(output, dim=1) : h(x)
            
            loss = torch.sum(losses) / len(losses)
        
        loss_reg = reg_loss_class(tab_mean_class)
        loss = loss + reg_term*loss_reg.data
        loss.requires_grad = True
        
        if train_loader.dataset.noise_mode == 'asymm':
            penalty = conf_penalty(output)
            L = loss + penalty
        else:
            L = loss
        
        L.backward()
        optimizer.step()
        
        ################## monitor losses  ####################################
        loss_per_batch.append(loss.item())
        ########################################################################

        # save accuracy:
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        acc_train_per_batch.append(100. * correct / ((batch_idx+1)*batch_size))

        if batch_idx % log_interval == 0:
            print('Epoch: {:>3}/{:>3} | [{:>5}/{:>5} ({:>3.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, epochs, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                       100. * correct / ((batch_idx + 1) * batch_size),
                optimizer.param_groups[0]['lr']))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [np.average(acc_train_per_batch)]  
    return (loss_per_epoch, acc_train_per_epoch)


def train_Sample_Boot(model, device, train_loader, optimizer, sample_ratio, epoch, epochs, TYPE, mixture, mixture_model, \
                            bmm_model_maxLoss, bmm_model_minLoss, batch_size, log_interval, reg_term=0):
    model.train()
    
    loss_per_batch = []
    acc_train_per_batch = []
    
    correct = 0
    conf_penalty = NegEntropy()
    
    for batch_idx, (Id, data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        output.detach_() 
        optimizer.zero_grad()
        
        output_mean = F.softmax(output, dim=1)
        tab_mean_class = torch.mean(output_mean, -2)
        
        if mixture == 'BMM':
            W = compute_probabilities_batch(data, target, model, mixture_model, bmm_model_maxLoss, bmm_model_minLoss, mixture=mixture)
            W = W.to(device)
            W[W <= 1e-4] = 1e-4
            W[W >= 1 - 1e-4] = 1 - 1e-4 # w
        
        elif mixture == 'GMM':
            W = compute_probabilities_batch(data, target, model, mixture_model, 0, 0, mixture=mixture)
        
        sample_size = int(sample_ratio*len(data))   
        indices = Weightedsampler(sample_size, W)
        Id_sample, data_sample, target_sample, W_sample = Id[indices], data[indices], target[indices], W[indices]
        
        output_sample = model(data_sample)
        
        if TYPE == 'Hard':
            output_sample = F.log_softmax(output_sample, dim=1) # h(x)
            z = torch.max(output_sample, dim=1)[1] # z
            losses = (1-W_sample) * F.nll_loss(output_sample, target_sample, reduction='none') + \
                     (  W_sample) * F.nll_loss(output_sample, z, reduction='none')
            
            loss = torch.sum(losses) / len(losses)

        elif TYPE == 'Soft':
            pred = F.log_softmax(output_sample, dim=1) # log(h(x))
            losses = (1-W_sample) * F.nll_loss(pred, target_sample, reduction='none') + \
                     (  W_sample) * -torch.sum(F.softmax(output_sample, dim=1) * pred, dim=1) # F.softmax(output, dim=1) : h(x)
            
            loss = torch.sum(losses) / len(losses)
        
        loss_reg = reg_loss_class(tab_mean_class)
        loss = loss + reg_term*loss_reg.data
        #loss.requires_grad = True
        
        if train_loader.dataset.noise_mode == 'asymm':
            penalty = conf_penalty(output)
            L = loss + penalty
        else:
            L = loss
        
        L.backward()
        optimizer.step()
        
        ################## monitor losses  ####################################
        loss_per_batch.append(loss.item())
        ########################################################################

        # save accuracy:
        pred = output_sample.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target_sample.view_as(pred)).sum().item()
        acc_train_per_batch.append(100. * correct / ((batch_idx+1)*batch_size))

        if batch_idx % log_interval == 0:
            print('Epoch: {:>3}/{:>3} | [{:>5}/{:>5} ({:>3.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, epochs, batch_idx * len(data_sample), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                       100. * correct / ((batch_idx + 1) * batch_size),
                optimizer.param_groups[0]['lr']))
        
    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [np.average(acc_train_per_batch)]

    return (loss_per_epoch, acc_train_per_epoch)

#patterns =['SC', 'PR', 'CC', 'LZ']
def trainCE(model, device, train_loader, train_loader_track, test_loader, optimizer, scheduler, epochs, NOISE_LEVEL,  batch_size, test_batch_size, log_interval, train_type='CE'):
    
    #bootstrap_ep_std = milestones[0] + 5 + 1 # the +1 is because the conditions are defined as ">" or "<" not ">="    
    # learning curve
    loss_train_per_epoch=[]
    acc_train_per_epoch_is=[]
    cont=0
    
    print('\t##### Doing standard training with cross-entropy loss #####')
    for epoch in range(1, epochs + 1):
        ######################## train ########################
        scheduler.step()
        
        loss_per_epoch, acc_train_per_epoch_i = train_CrossEntropy(model, device, train_loader, optimizer, epoch, epochs, batch_size, log_interval)
        #######################################################
        
        
        ######################## write logs ########################
        loss_train_per_epoch.append(loss_per_epoch[0])
        acc_train_per_epoch_is.append(acc_train_per_epoch_i[0])       
        ############################################################
        
        
        ######################## test ########################
        loss_per_epoch_test, acc_val_per_epoch_i = test_cleaning(model, device, test_loader, test_batch_size, log_interval)
        #######################################################
        
        ######################## model save ########################
        if epoch == 1:
            best_acc_val = acc_val_per_epoch_i[-1]
            snapBest = 'best_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_%.2f_bestAccVal_%.5f' % (
                epoch, loss_per_epoch_test[-1], acc_val_per_epoch_i[-1], NOISE_LEVEL, best_acc_val)
            directory = './{}'.format(train_loader.dataset.noise_mode) # noise type / train_type / pattern
            os.makedirs(directory, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(directory, snapBest + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(directory, 'opt_' + snapBest + '.pth'))
        else:
            if acc_val_per_epoch_i[-1] > best_acc_val:
                best_acc_val = acc_val_per_epoch_i[-1]

                if cont>0:
                    try:
                        os.remove(os.path.join(directory, 'opt_' + snapBest + '.pth'))
                        os.remove(os.path.join(directory, snapBest + '.pth'))
                    except OSError:
                        pass
                snapBest = 'best_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_%.2f_bestAccVal_%.5f' % (
                    epoch, loss_per_epoch_test[-1], acc_val_per_epoch_i[-1], NOISE_LEVEL, best_acc_val)
                torch.save(model.state_dict(), os.path.join(directory, snapBest + '.pth'))
                print('└ save the best model in {}/{}\n'.format(directory, snapBest+'.pth')) 
                #torch.save(optimizer.state_dict(), os.path.join(directory, 'opt_' + snapBest + '.pth'))

        cont+=1

        #if epoch == epochs:
        #    snapLast = 'last_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_%.2f_bestValLoss_%.5f' % (
        #        epoch, loss_per_epoch_test[-1], acc_val_per_epoch_i[-1], NOISE_LEVEL, best_acc_val)
        #    torch.save(model.state_dict(), os.path.join(directory, snapLast + '.pth'))
        #    torch.save(optimizer.state_dict(), os.path.join(directory, 'opt_' + snapLast + '.pth'))
       ############################################################
    
    # plotting
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    axs[0].plot(loss_train_per_epoch, color='b')
    axs[0].set(xlabel='epochs', ylabel='loss')
    axs[0].set_title('Loss per epoch', fontsize=16)

    axs[1].plot(acc_train_per_epoch_is, color='g')
    axs[1].set(xlabel='epochs', ylabel='acc')
    axs[1].set_title('Accuracy per epoch', fontsize=16)
    fig.tight_layout()
    plt.show()
    return model, (loss_train_per_epoch, acc_train_per_epoch_is)


def sample_boot(model, device, train_loader, train_loader_track, test_loader, learning_rate, weight_decay, epochs, restart, mixture, max_iters, NOISE_LEVEL, sample_ratio, batch_size, test_batch_size, log_interval, milestones, reg_term=0, TYPE='Hard', train_type='SampleBoot'):
    
    #bootstrap_ep_std = milestones[0] + 5 + 1 # the +1 is because the conditions are defined as ">" or "<" not ">="
    warmup = milestones[0]
    mixture_model=bmm_model_maxLoss=bmm_model_minLoss=cont=k = 0
    
    # learning curve
    loss_train_per_epoch=[]
    acc_train_per_epoch_is=[]
    
    loss_test_per_epoch=[]
    acc_val_per_epoch_is=[]
    
    Recall_clean_track=[]
    Recall_noise_track=[]
    
    for r in range(restart+1):
        
        optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay, lr=learning_rate)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        print('\n*---------------------* {} run *----------------------*'.format(r+1))
        
        for epoch in range(1, epochs + 1):
            # train
            scheduler.step()

            if epoch <= warmup:
                ### Standard CE training (without mixup) ###
                print('\n\t##### Doing standard training with cross-entropy loss #####')
                loss_per_epoch, acc_train_per_epoch_i = train_CrossEntropy(model, device, train_loader,
                                                                           optimizer, epoch, epochs, batch_size, log_interval)


            else:
                if mixture=='BMM':
                    if TYPE == "Hard":
                        print("\n\t##### Doing HARD sample BETA-boot after epoch {}__{} #####".format(warmup+1, mixture))
                        loss_per_epoch, acc_train_per_epoch_i = train_Sample_Boot(
                                                                                  model, device, train_loader,
                                                                                  optimizer, sample_ratio, epoch, epochs, TYPE, mixture, 
                                                                                  mixture_model, bmm_model_maxLoss, bmm_model_minLoss, 
                                                                                  batch_size, log_interval, reg_term)
                    
                    elif TYPE == "Soft":
                        print("\n\t##### Doing SOFT sample BETA-boot after epoch {}__{} #####".format(warmup, mixture))
                        loss_per_epoch, acc_train_per_epoch_i = train_Sample_Boot(
                                                                                  model, device, train_loader,
                                                                                  optimizer, sample_ratio, epoch, epochs, TYPE, mixture, 
                                                                                  mixture_model, bmm_model_maxLoss, bmm_model_minLoss,
                                                                                  batch_size, log_interval, reg_term)
                    
                elif mixture=='GMM':
                    if TYPE == "Hard":
                        print("\n\t##### Doing HARD BETA-boot after epoch {}__{} #####".format(warmup, mixture))
                        loss_per_epoch, acc_train_per_epoch_i = train_Sample_Boot(
                                                                                  model, device, train_loader,
                                                                                  optimizer, sample_ratio, epoch, epochs, TYPE, mixture,
                                                                                  mixture_model, bmm_model_maxLoss, bmm_model_minLoss,
                                                                                  batch_size, log_interval, reg_term)
                    elif TYPE == "Soft":
                        print("\n\t##### Doing SOFT BETA-boot after epoch {}__{} #####".format(warmup, mixture))
                        loss_per_epoch, acc_train_per_epoch_i = train_Sample_Boot(
                                                                                  model, device, train_loader,
                                                                                  optimizer, sample_ratio, epoch, epochs, TYPE, mixture,
                                                                                  mixture_model, bmm_model_maxLoss, bmm_model_minLoss, 
                                                                                  batch_size, log_interval, reg_term)
            
            ### Training tracking loss (BMM Modeling) 
            epoch_losses_train, epoch_probs_train, argmaxXentropy_train, mixture_model, bmm_model_maxLoss, bmm_model_minLoss = \
                track_training_loss(model, device, train_loader_track,
                                    epoch, mixture, max_iters, mixture_model, bmm_model_maxLoss, bmm_model_minLoss)

            # write logs
            loss_train_per_epoch.append(loss_per_epoch[0])
            acc_train_per_epoch_is.append(acc_train_per_epoch_i[0])            
            
            # test
            loss_per_epoch_test, acc_val_per_epoch_i = test_cleaning(model, device, test_loader,
                                                                     test_batch_size, log_interval)
            
            loss_test_per_epoch.append(loss_per_epoch_test)
            acc_val_per_epoch_is.append(acc_val_per_epoch_i)
            
            
            if epoch == 1:
                best_acc_val = acc_val_per_epoch_i[-1]
                snapBest = 'best_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_%.2f_bestAccVal_%.5f' % (
                    epoch, loss_per_epoch_test[-1], acc_val_per_epoch_i[-1], NOISE_LEVEL, best_acc_val)
                directory = './{}/{}'.format(train_loader.dataset.noise_mode, TYPE) # noise type / Hard or Soft / pattern
                os.makedirs(directory, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(directory, snapBest + '.pth'))
                print('└ save the best model in {}/{}\n'.format(directory, snapBest+'.pth')) 
                #torch.save(optimizer.state_dict(), os.path.join(directory, 'opt_' + snapBest + '.pth'))
                
            else:
                if acc_val_per_epoch_i[-1] > best_acc_val:
                    best_acc_val = acc_val_per_epoch_i[-1]

                    if cont>0:
                        try:
                            os.remove(os.path.join(directory, 'opt_' + snapBest + '.pth'))
                            os.remove(os.path.join(directory, snapBest + '.pth'))
                        except OSError:
                            pass
                    snapBest = '%d_%s_best_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_%.2f_bestAccVal_%.5f' % (
                        r+1, mixture, epoch, loss_per_epoch_test[-1], acc_val_per_epoch_i[-1], NOISE_LEVEL, best_acc_val)
                    torch.save(model.state_dict(), os.path.join(directory, snapBest + '.pth'))
                    #torch.save(optimizer.state_dict(), os.path.join(directory, 'opt_' + snapBest + '.pth'))

            cont+=1

            if epoch == epochs:
                snapLast = '%s_last_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_%.2f_bestValLoss_%.5f' % (
                    mixture, epoch, loss_per_epoch_test[-1], acc_val_per_epoch_i[-1], NOISE_LEVEL, best_acc_val)
                #torch.save(model.state_dict(), os.path.join(directory, snapLast + '.pth'))
                #torch.save(optimizer.state_dict(), os.path.join(directory, 'opt_' + snapLast + '.pth'))

        font = {'size': 14}
        matplotlib.rc('font', **font)

    
        # plotting
        fig, axs = plt.subplots(2, 2, figsize=(15, 6))
        axs[0, 0].plot(loss_train_per_epoch, color='m')
        axs[0, 0].set(xlabel='epochs', ylabel='loss')
        axs[0, 0].set_title('Train Loss per epoch', fontsize=16)

        axs[0, 1].plot(acc_train_per_epoch_is, color='g')
        axs[0, 1].set(xlabel='epochs', ylabel='acc')
        axs[0, 1].set_title('Train Accuracy per epoch', fontsize=16)

        axs[1, 0].plot(loss_test_per_epoch, color='r')
        axs[1, 0].set(xlabel='epochs', ylabel='loss')
        axs[1, 0].set_title('Val Loss per epoch', fontsize=16)

        axs[1, 1].plot(acc_val_per_epoch_is, color='b')
        axs[1, 1].set(xlabel='epochs', ylabel='acc')
        axs[1, 1].set_title('Val Accuracy per epoch', fontsize=16)
        fig.tight_layout()
        plt.show()

    
    return model, (loss_train_per_epoch, acc_train_per_epoch_is), directory+'/'+snapBest+'.pth'


def sample_boot(model, device, train_loader, train_loader_track, test_loader, learning_rate, weight_decay,
                epochs, restart, mixture, max_iters, NOISE_LEVEL, sample_ratio, batch_size, test_batch_size,
                log_interval, milestones, reg_term=0, TYPE='Hard', train_type='SampleBoot'):
    # bootstrap_ep_std = milestones[0] + 5 + 1 # the +1 is because the conditions are defined as ">" or "<" not ">="
    warmup = milestones[0]
    mixture_model = bmm_model_maxLoss = bmm_model_minLoss = cont = k = 0

    # learning curve
    loss_train_per_epoch = []
    acc_train_per_epoch_is = []

    loss_test_per_epoch = []
    acc_val_per_epoch_is = []

    Recall_clean_track = []
    Recall_noise_track = []

    for r in range(restart + 1):

        optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay, lr=learning_rate)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        print('\n*---------------------* {} run *----------------------*'.format(r + 1))

        for epoch in range(1, epochs + 1):
            # train
            scheduler.step()

            if epoch <= warmup:
                ### Standard CE training (without mixup) ###
                print('\n\t##### Doing standard training with cross-entropy loss #####')
                loss_per_epoch, acc_train_per_epoch_i = train_CrossEntropy(model, device, train_loader,
                                                                           optimizer, epoch, epochs, batch_size,
                                                                           log_interval)


            else:
                if mixture == 'BMM':
                    if TYPE == "Hard":
                        print("\n\t##### Doing HARD sample BETA-boot after epoch {}__{} #####".format(warmup + 1,
                                                                                                      mixture))
                        loss_per_epoch, acc_train_per_epoch_i = train_Sample_Boot(
                            model, device, train_loader,
                            optimizer, sample_ratio, epoch, epochs, TYPE, mixture,
                            mixture_model, bmm_model_maxLoss, bmm_model_minLoss,
                            batch_size, log_interval, reg_term)

                    elif TYPE == "Soft":
                        print("\n\t##### Doing SOFT sample BETA-boot after epoch {}__{} #####".format(warmup, mixture))
                        loss_per_epoch, acc_train_per_epoch_i = train_Sample_Boot(
                            model, device, train_loader,
                            optimizer, sample_ratio, epoch, epochs, TYPE, mixture,
                            mixture_model, bmm_model_maxLoss, bmm_model_minLoss,
                            batch_size, log_interval, reg_term)

                elif mixture == 'GMM':
                    if TYPE == "Hard":
                        print("\n\t##### Doing HARD BETA-boot after epoch {}__{} #####".format(warmup, mixture))
                        loss_per_epoch, acc_train_per_epoch_i = train_Sample_Boot(
                            model, device, train_loader,
                            optimizer, sample_ratio, epoch, epochs, TYPE, mixture,
                            mixture_model, bmm_model_maxLoss, bmm_model_minLoss,
                            batch_size, log_interval, reg_term)
                    elif TYPE == "Soft":
                        print("\n\t##### Doing SOFT BETA-boot after epoch {}__{} #####".format(warmup, mixture))
                        loss_per_epoch, acc_train_per_epoch_i = train_Sample_Boot(
                            model, device, train_loader,
                            optimizer, sample_ratio, epoch, epochs, TYPE, mixture,
                            mixture_model, bmm_model_maxLoss, bmm_model_minLoss,
                            batch_size, log_interval, reg_term)

            ### Training tracking loss (BMM Modeling)
            epoch_losses_train, epoch_probs_train, argmaxXentropy_train, mixture_model, bmm_model_maxLoss, bmm_model_minLoss = \
                track_training_loss(model, device, train_loader_track,
                                    epoch, mixture, max_iters, mixture_model, bmm_model_maxLoss, bmm_model_minLoss)

            # write logs
            loss_train_per_epoch.append(loss_per_epoch[0])
            acc_train_per_epoch_is.append(acc_train_per_epoch_i[0])

            # test
            loss_per_epoch_test, acc_val_per_epoch_i = test_cleaning(model, device, test_loader,
                                                                     test_batch_size, log_interval)

            loss_test_per_epoch.append(loss_per_epoch_test)
            acc_val_per_epoch_is.append(acc_val_per_epoch_i)

            if epoch == 1:
                best_acc_val = acc_val_per_epoch_i[-1]
                snapBest = 'best_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_%.2f_bestAccVal_%.5f' % (
                    epoch, loss_per_epoch_test[-1], acc_val_per_epoch_i[-1], NOISE_LEVEL, best_acc_val)
                directory = './{}/{}'.format(train_loader.dataset.noise_mode, TYPE,
                                                )  # noise type / Hard or Soft / pattern
                os.makedirs(directory, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(directory, snapBest + '.pth'))
                print('└ save the best model in {}/{}\n'.format(directory, snapBest + '.pth'))
                # torch.save(optimizer.state_dict(), os.path.join(directory, 'opt_' + snapBest + '.pth'))

            else:
                if acc_val_per_epoch_i[-1] > best_acc_val:
                    best_acc_val = acc_val_per_epoch_i[-1]

                    if cont > 0:
                        try:
                            os.remove(os.path.join(directory, 'opt_' + snapBest + '.pth'))
                            os.remove(os.path.join(directory, snapBest + '.pth'))
                        except OSError:
                            pass
                    snapBest = '%d_%s_best_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_%.2f_bestAccVal_%.5f' % (
                        r + 1, mixture, epoch, loss_per_epoch_test[-1], acc_val_per_epoch_i[-1], NOISE_LEVEL,
                        best_acc_val)
                    torch.save(model.state_dict(), os.path.join(directory, snapBest + '.pth'))
                    # torch.save(optimizer.state_dict(), os.path.join(directory, 'opt_' + snapBest + '.pth'))

            cont += 1

            if epoch == epochs:
                snapLast = '%s_last_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_%.2f_bestValLoss_%.5f' % (
                    mixture, epoch, loss_per_epoch_test[-1], acc_val_per_epoch_i[-1], NOISE_LEVEL, best_acc_val)
                # torch.save(model.state_dict(), os.path.join(directory, snapLast + '.pth'))
                # torch.save(optimizer.state_dict(), os.path.join(directory, 'opt_' + snapLast + '.pth'))

        font = {'size': 14}
        matplotlib.rc('font', **font)

        # plotting
        fig, axs = plt.subplots(2, 2, figsize=(15, 6))
        axs[0, 0].plot(loss_train_per_epoch, color='m')
        axs[0, 0].set(xlabel='epochs', ylabel='loss')
        axs[0, 0].set_title('Train Loss per epoch', fontsize=16)

        axs[0, 1].plot(acc_train_per_epoch_is, color='g')
        axs[0, 1].set(xlabel='epochs', ylabel='acc')
        axs[0, 1].set_title('Train Accuracy per epoch', fontsize=16)

        axs[1, 0].plot(loss_test_per_epoch, color='r')
        axs[1, 0].set(xlabel='epochs', ylabel='loss')
        axs[1, 0].set_title('Val Loss per epoch', fontsize=16)

        axs[1, 1].plot(acc_val_per_epoch_is, color='b')
        axs[1, 1].set(xlabel='epochs', ylabel='acc')
        axs[1, 1].set_title('Val Accuracy per epoch', fontsize=16)
        fig.tight_layout()
        plt.show()

    return model, (loss_train_per_epoch, acc_train_per_epoch_is), directory + '/' + snapBest + '.pth'