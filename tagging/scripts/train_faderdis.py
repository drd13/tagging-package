"""Script for training the fadder method without the metallicity"""


import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import argparse
import pickle

sys.path.insert(0,'/share/splinter/ddm/taggingClean/')



from src.datasets import ApogeeDataset
from src.networks import ConditioningAutoencoder,Embedding_Decoder,Feedforward
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

############################################

parser = argparse.ArgumentParser()
parser.add_argument("--data_file", type=str, default="train/spectra_noiseless.pd", help="file to used for training")
parser.add_argument("--n_batch", type=int, default=256, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00001, help="learning rate")
parser.add_argument("--n_z", type=int, default=20, help="dimensionality of the latent space")
parser.add_argument("--n_bins", type=int, default=7751, help="size of each image dimension")
parser.add_argument("--n_conditioned", type=int, default=2, help="number of parameters conditioned")
parser.add_argument("--n_cat", type=int, default=30, help="discretization used by fadder network")
parser.add_argument("--loss_ratio", type=float, default=0.00001, help="discretization used by fadder network")
parser.add_argument("--noise", type=float, default=0.0, help="signal to noise ratio")
opt = parser.parse_args()

print("the hyperparameters for this run are:")
print(opt)
with open("parameters.p","wb") as f: #save hyperparameters to make them accesibl
    pickle.dump(opt,f)



###############################################

"""Load and batch the dataset"""

data = pd.read_pickle(opt.data_file)
dataset = ApogeeDataset(data[:50000],opt.n_bins)
loader = torch.utils.data.DataLoader(dataset = dataset,
                                     batch_size = opt.n_batch,
                                     shuffle = True,
                                     drop_last=True)

####################################################

"""Initialize the neural networks"""

encoder = Feedforward([opt.n_bins+opt.n_conditioned,2048,512,128,32,opt.n_z],activation=nn.SELU()).to(device)
decoder = Feedforward([opt.n_z+opt.n_conditioned,512,2048,8192,opt.n_bins],activation=nn.SELU()).to(device)
conditioning_autoencoder = ConditioningAutoencoder(encoder,decoder,n_bins=opt.n_bins,n_embedding=0).to(device)
#conditioning_autoencoder = torch.load("../../feed/trueCont/exp1/feedN7214I1700") 
#conditioning_autoenocoder = torch.load("adN7214I560")

pred_u0_given_v = Feedforward([opt.n_z+1,512,256,opt.n_cat],activation=nn.SELU()).to(device)
pred_u1_given_v = Feedforward([opt.n_z+1,512,256,opt.n_cat],activation=nn.SELU()).to(device)


###################################################

"""Initialize the network losses and optimizers"""

loss = nn.MSELoss()
loss2 = nn.CrossEntropyLoss()
lr2 = 0.00001
optimizer_autoencoder = torch.optim.Adam(conditioning_autoencoder.parameters(), lr=opt.lr)
optimizer_u0 = torch.optim.Adam(pred_u0_given_v.parameters(), lr=opt.lr)
optimizer_u1 = torch.optim.Adam(pred_u1_given_v.parameters(), lr=opt.lr)


#######################Train#########################
zeros = torch.full((opt.n_batch,2), 0.0, device=device)
ones = torch.full((opt.n_batch,2),1.0,device=device)

noise_matrix = torch.empty(50000,opt.n_bins).normal_(mean=0,std=1/opt.noise).to(device)*4 #We initialize one noisy version of every datapoint and always use the same noise. This was found to work better (but not fully understood)

for i in range(20000):
    if i%200==0:
        """We save the neural network every x iterations"""
        torch.save(conditioning_autoencoder, "adN7214I"+str(i)) 
        torch.save(pred_u0_given_v, "u0I"+str(i)) 
        torch.save(pred_u1_given_v, "u1I"+str(i))
    for j,(x,u,v,idx) in enumerate(loader):
        """Training loop"""
        if opt.noise!=0:
            noise = noise_matrix[idx]
            x+=noise

        u_cat = ((u+1)*opt.n_cat/2).long()
        u_cat[u_cat==opt.n_cat]=opt.n_cat-1

        optimizer_autoencoder.zero_grad()
        x_pred,z = conditioning_autoencoder(x,u[:,0:2].detach())

        err_pred = loss(x_pred,x)  

        z0 = torch.cat((z,u[:,1:2]),1)
        z1 = torch.cat((z,u[:,0:1]),1)
        u0_pred = pred_u0_given_v(z0)  
        u1_pred = pred_u1_given_v(z1)  
        err_u0 = loss2(u0_pred,u_cat[:,0])
        err_u1 = loss2(u1_pred,u_cat[:,1])
        err_tot = err_pred-opt.loss_ratio*err_u0-opt.loss_ratio*err_u1 #agrregated loss

        err_tot.backward(retain_graph=True)
        optimizer_autoencoder.step()
        optimizer_u0.zero_grad()
        err_u0.backward(retain_graph=True)
        optimizer_u0.step()
        optimizer_u1.zero_grad()
        err_u1.backward()
        optimizer_u1.step()
        if j%10==0:
            print("tot:{},err:{},err_u0:{},er_u1:{}".format(err_tot,err_pred,err_u0,err_u1))
