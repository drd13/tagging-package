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
from scipy import spatial
import sys
import argparse
import pickle



from tagging.src.datasets import ApogeeDataset
from tagging.src.networks import ConditioningAutoencoder,Embedding_Decoder,Feedforward

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


############################################

parser = argparse.ArgumentParser()
parser.add_argument("--data_file", type=str, default="train/spectra_noiseless.pd", help="file containing data")
parser.add_argument("--model_file", type=str, default="../wganI2000", help="file containing model")
parser.add_argument("--n_batch", type=int, default=250, help="size of the batches")
parser.add_argument("--n_bins", type=int, default=7751, help="size of each image dimension")
parser.add_argument("--n_z", type=int, default=20, help="number of latent dimensions")
parser.add_argument("--n_conditioned", type=int, default=3, help="number of parameters conditioned")
parser.add_argument("--savepath", type=str, help="where to save the pickled files containing results")
opt = parser.parse_args()

print("the hyperparameters for this run are:")
print(opt)
with open("parameters.p","wb") as f: #save hyperparameters to make them accesibl
    pickle.dump(opt,f)

################################################


data = pd.read_pickle(opt.data_file)


dataset = ApogeeDataset(data,opt.n_bins)
evaluation_loader = torch.utils.data.DataLoader(dataset = dataset,
                                     batch_size = opt.n_batch,
                                     shuffle = False,
                                     drop_last=True)



encoder = Feedforward([opt.n_bins+opt.n_conditioned,2048,512,128,32,opt.n_z],activation=nn.SELU()).to(device)
decoder = Feedforward([opt.n_z+opt.n_conditioned,512,2048,8192,opt.n_bins],activation=nn.SELU()).to(device)
conditioning_autoencoder = ConditioningAutoencoder(encoder,decoder,n_bins=opt.n_bins).to(device)
conditioning_autoencoder.load_state_dict(torch.load(opt.model_file))


##################################################################
#run throught the data and get all the latents and stellar spectra
dataset_latents= []
dataset_xs=[]
for j,(z,u,v,idx) in enumerate(evaluation_loader):
  print(j)
  z_pred,z_latent = conditioning_autoencoder(z,u[:,0:opt.n_conditioned])
  if j==0:
    dataset_latents = z_latent.detach().cpu().numpy()
    dataset_xs = z.detach().cpu().numpy()

   
  else:
    dataset_latents = np.append(dataset_latents,z_latent.detach().cpu().numpy(),axis=0)
    dataset_xs = np.append(dataset_xs,z.detach().cpu().numpy(),axis=0)

###################################################################
#find the closest stellar siblings

latent_ranking=[]
latent_tree = spatial.KDTree(dataset_latents)
no_siblings = 0

for i in range(25000):
    d,idx2 = latent_tree.query(dataset_latents[i],p=2,k=50000)
    if 25000+i in idx2:
        pos = np.where(idx2==25000+i)[0][0]
        print("{} is:{}".format(i,pos))
        latent_ranking.append(pos)
        if pos==1:
            no_siblings+=1
    else:
        print("{} is:{}".format(i,50000))
        latent_ranking.append(50000)

print("stars with no siblings:{}".format(no_siblings))

with open(opt.savepath,"wb") as f:
    pickle.dump(latent_ranking,f) 
