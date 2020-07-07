import torch 
import torchvision
import numpy as np
import pickle

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
    nlam = 8575 
    start_wl = 4.179 
    diff_wl = 6e-06 
    val = diff_wl*(nlam) + start_wl  
    wl_full_log = np.arange(start_wl,val, diff_wl) 
    wl_full = [10**aval for aval in wl_full_log] 
    xdata = np.array(wl_full)  
    return xdata
