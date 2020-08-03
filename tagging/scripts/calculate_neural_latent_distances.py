"""Provided a dataset, this script can be used to calculate a RMS distance between twins in the dataset and between non twins"""

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
from tagging.src.utils import load_model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


############################################

parser = argparse.ArgumentParser()
parser.add_argument("--data_file", type=str, default="train/spectra_noiseless.pd", help="file containing data")
parser.add_argument("--model_file", type=str, default="../wganI2000", help="file containing model")
parser.add_argument("--n_batch", type=int, default=250, help="size of the batches")
parser.add_argument("--n_bins", type=int, default=7751, help="number of bins that the model uses")
parser.add_argument("--n_z", type=int, default=20, help="number of latent dimensions")
parser.add_argument("--n_conditioned", type=int, default=3, help="If 2 condition on temperature and surface gravity. If 3 also condition on metallicity")
parser.add_argument("--savepath", type=str, help="where to save the pickled files containing results")

opt = parser.parse_args()

data = pd.read_pickle(opt.data_file)


dataset = ApogeeDataset(data,opt.n_bins)
evaluation_loader = torch.utils.data.DataLoader(dataset = dataset,
                                     batch_size = opt.n_batch,
                                     shuffle = False,
                                     drop_last=True)



conditioning_autoencoder = load_model(opt.model_file)

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



sibling_rms = []
random_rms = []
for i in range(25000):
    current = dataset_latents[i]
    sibling = dataset_latents[i+25000]
    random = dataset_latents[i+1]
    dist_siblings = np.linalg.norm(current-sibling)
    dist_randoms  = np.linalg.norm(current-random)
    sibling_rms.append(dist_siblings)
    random_rms.append(dist_randoms)

distances = {"siblings":sibling_rms,
             "randoms":random_rms}

with open(opt.savepath,"wb") as f:
    pickle.dump(distances,f)
