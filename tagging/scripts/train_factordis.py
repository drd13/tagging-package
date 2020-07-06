import argparse
import os
import numpy as np
import math
import sys
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import pickle
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import Dataset,DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import argparse


from tagging.src.datasetsOld import ApogeeDataset
from tagging.src.networks import ConditioningAutoencoder,Embedding_Decoder,Feedforward
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False

############################################

parser = argparse.ArgumentParser()
parser.add_argument("--data_file", type=str, default="train/spectra_noiseless.pd", help="file to used for training")
parser.add_argument("--n_batch", type=int, default=256, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00001, help="learning rate")
parser.add_argument("--n_z", type=int, default=20, help="dimensionality of the latent space")
parser.add_argument("--n_bins", type=int, default=7751, help="size of each image dimension")
parser.add_argument("--n_conditioned", type=int, default=2, help="number of parameters conditioned")
parser.add_argument("--lambda_fact", type=float, default=0.0001, help="weight used for the factorizing loss")
parser.add_argument("--noise", type=float, default=0.0, help="signal to noise ratio")
parser.add_argument("--disentangle_z",type=bool,default=False, help="whether to include the metallicity z into the disentangled parameters or not")
opt = parser.parse_args()

n_embedding = 0 #this isn't explicity being used but I can't be bothered to change my code
print("the hyperparameters for this run are:")
print(opt)
with open("parameters.p","wb") as f: #save hyperparameters to make them accesibl
    pickle.dump(opt,f)

################################################

data = pd.read_pickle(opt.data_file)
#data = pd.read_pickle("/mnt/home/dmijolla/taggingPaper/data/final/{}".format(opt.data_file))
dataset = ApogeeDataset(data[:50000],opt.n_bins)
loader = torch.utils.data.DataLoader(dataset = dataset,
                                     batch_size = opt.n_batch,
                                     shuffle = True,
                                     drop_last=True)

####################################################

encoder = Feedforward([opt.n_bins+opt.n_conditioned,2048,512,128,32,opt.n_z],activation=nn.SELU()).to(device)
decoder = Feedforward([opt.n_z+opt.n_conditioned,512,2048,8192,opt.n_bins],activation=nn.SELU()).to(device)
conditioning_autoencoder = ConditioningAutoencoder(encoder,decoder,n_bins=opt.n_bins,n_embedding = n_embedding).to(device)

discriminator = Feedforward([opt.n_bins+opt.n_z+opt.n_conditioned,4096,1024,512,128,32,1],activation=nn.SELU()).to(device)

#conditioning_autoencoder = torch.load("wganI4000") 
#discriminator = torch.load("discI4000") 

loss = nn.MSELoss()

# Loss weight for gradient penalty
lambda_gp = 10
n_critic = 10
# Initialize generator and discriminator



# Optimizers
optimizer_G = torch.optim.Adam(conditioning_autoencoder.parameters(), lr=opt.lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty




# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(10000):
    if epoch%100==0:
        torch.save(conditioning_autoencoder, "wganI"+str(epoch)) 
        torch.save(discriminator, "discI"+str(epoch)) 
        torch.save(optimizer_D, "optD"+str(epoch)) 
        torch.save(optimizer_G, "optG"+str(epoch)) 

    for i, (x,u,v,idx) in enumerate(loader):
        if opt.noise!=0:
            #print("adding noise")
            noise = x.data.new(x.size()).normal_(0.,1/opt.noise)*4  #the times 4 is too account for the scaling/normalization that has already been done.
            x+=noise
        optimizer_G.zero_grad()
        n_perm=torch.randperm(opt.n_batch)
        u_perm = u[n_perm]
        n_perm2=torch.randperm(opt.n_batch)

        if opt.disentangle_z:
            x_pred,z = conditioning_autoencoder(x,u[:,0:3])
            x_perm,_ = conditioning_autoencoder(z,u_perm[:,0:3],train_encoder=False)
        else:
            x_pred,z = conditioning_autoencoder(x,u[:,0:2])
            x_perm,_ = conditioning_autoencoder(z,u_perm[:,0:2],train_encoder=False)



        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        if opt.disentangle_z:
            fake = torch.cat((x_perm,z,u_perm[:,0:3]),1)
            real = torch.cat((x,z,u[:,0:3]),1)
        else:
            fake = torch.cat((x_perm,z,u_perm[:,0:2]),1)
            real = torch.cat((x,z,u[:,0:2]),1)

        # Real images
        real_validity = discriminator(real)
        # Fake images
        fake_validity = discriminator(fake)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real.data, fake.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

        d_loss.backward(retain_graph=True)
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake)
            real_validity = discriminator(real)

            err_pred = loss(x_pred,x)

            g_loss = err_pred-opt.lambda_fact*(torch.mean(fake_validity)-torch.mean(real_validity))


            g_loss.backward()
            optimizer_G.step()
            if i%30*n_critic ==0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D: %f] [G: %f] [R: %f]"
                    % (epoch, 100, i, len(loader), d_loss.item(), torch.mean(fake_validity).item(),err_pred.item())
                )

        
                batches_done += n_critic








