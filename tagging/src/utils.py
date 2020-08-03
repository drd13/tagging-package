import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os
import pickle
import collections


from tagging.src.networks import ConditioningAutoencoder,Embedding_Decoder,Feedforward
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def invert_x(x):
    """undo the preprocessing used by the neural network"""
    return (x+3.5)/4


def get_batch(j,n_batch,dataset):
  """"Return batch as a specific index
  
  Parameters
  ----------
  j: int
    index to start batch at
  n_batch: int
    number of datapoints to include in batch
  dataset: pytorch dataset
    dataset to extract
    
  """
  idx1 = []
  for i in range(n_batch):
    x,u,v,idx = dataset[i+j]
    x = x.unsqueeze(0)
    u = u.unsqueeze(0)
    v = v.unsqueeze(0)
    idx1.append(idx)
    if i==0:
      x1=x
      u1=u
      v1=v

    else:
      x1 = torch.cat((x1,x))
      u1 = torch.cat((u1,u))
      v1 = torch.cat((v1,v))
  return x1,u1,v1,idx1

def get_xdata():
    """returns an array containing the wavelength of each bin"""
    """
    ### old way of getting x
    nlam = 8575 
    start_wl = 4.179 
    diff_wl = 6e-06 
    val = diff_wl*(nlam) + start_wl  
    wl_full_log = np.arange(start_wl,val, diff_wl) 
    wl_full = [10**aval for aval in wl_full_log] 
    xdata = np.array(wl_full)"""
    pickled_spectra = os.path.join(os.path.dirname("/share/splinter/ddm/taggingProject/taggingRepo/tagging/src/utils.py"),"x_spectra.p")
    with open(pickled_spectra,"rb") as f:
        xdata = pickle.load(f)
    return xdata


def load_model(model_path,n_bins=7751,n_conditioned=3,n_z=20):
    """loads a model from the path"""
    conditioning_autoencoder = torch.load(model_path)
    if type(conditioning_autoencoder) == collections.OrderedDict:
        encoder = Feedforward([n_bins+n_conditioned,2048,512,128,32,n_z],activation=nn.SELU()).to(device)
        decoder = Feedforward([n_z+n_conditioned,512,2048,8192,n_bins],activation=nn.SELU()).to(device)
        conditioning_autoencoder = ConditioningAutoencoder(encoder,decoder,n_bins=n_bins).to(device)
        weights  = torch.load(model_path)
        try:
            del weights['w.weight']
        except:
            pass
        conditioning_autoencoder.load_state_dict(weights)
    return conditioning_autoencoder

