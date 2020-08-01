import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import pickle


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ApogeeDataset(Dataset):
    def __init__(self,data,n_bins):
        print("inside")
        print(len(data))
        self.n_bins = n_bins
        self.dataset = data
        print(len(self.dataset))
        self.parameters = self.load_params()
        self.abundances = self.load_abunds()
        print("made it this far")
        self.spectra = data["spectra"]
        print("made it to the end far")

    def load_params(self):
        parameters=[]
        for i in range(len(self.dataset)):
            parameters.append(self.dataset["params"][i])
        parameters = np.array(parameters)
        for i in range(len(parameters[0])):
            parameters[:,i]=(parameters[:,i]-min(parameters[:,i]))/(max(parameters[:,i])-min(parameters[:,i]))
        parameters = np.nan_to_num(parameters)
        return parameters
      
            
    def load_abunds(self):
        print("dataset size {}".format(len(self.dataset)))  
        abundances=[]
        for i in range(len(self.dataset)):
            abundances.append(self.dataset["abundances"][i])
        abundances = np.array(abundances)
        abundances[abundances == -9.9990e+03] = -3
        for i in range(len(abundances[0])):
            abundances[:,i]=(abundances[:,i]-min(abundances[:,i]))/(max(abundances[:,i])-min(abundances[:,i]))
        abundances = np.nan_to_num(abundances)
        return abundances
        

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        spectra = self.spectra[idx]#[np.nonzero(self.spectra[idx])[0]]
        z = torch.tensor(spectra[0:0+self.n_bins]*4-3.5,device=device)
        u = torch.tensor(self.parameters[idx]*2-1,device=device)
        v = torch.tensor(self.abundances[idx]*2-1,device=device)
        return z.float(),u.float(),v.float(),idx
    
    
    
    
class ToyDataset(Dataset):
    def __init__(self,n_datapoints,n_signals,n_bins,fraction_signal):
        """
        n_datapoints: int
            number of datapoints in dataset
        n_signals: int
            number of laten variables to include in the signal
        n_bins: int
            number of bins in the latent signal
        fraction_signal: int
            fractions of bins containing each signal"""
            
        self.n_datapoints = n_datapoints
        self.n_signals = n_signals
        self.n_bins = n_bins
        self.fraction_signal = fraction_signal
    
        self.basis = self.generate_signal_basis(self.n_signals,self.n_bins,self.fraction_signal)
        self.amplitudes = self.generate_amplitudes_dataset(self.n_signals,self.n_datapoints)
        self.dataset = self.generate_dataset(self.basis,self.amplitudes)
        self.dataset = torch.FloatTensor(self.dataset)

        
    def generate_signal_basis(self,n_signals,n_bins,fraction_signal):
        """returns an np.array of shape n_signals*n_bins where every row is the basis of its respective signal"""
        return (np.random.random_sample(size=(n_signals,n_bins))<fraction_signal).astype(float)
    
    def generate_amplitudes_dataset(self,n_signals,n_datapoints):
        """generate the amplitude of each observation signal bin
        returns an array of shape n_signals*n_datapoints"""
        return np.random.normal(0,0.3,size= (n_signals,n_datapoints))
    
    def generate_dataset(self,basis,amplitudes):
        print(basis.shape)
        print(amplitudes.shape)
        return np.matmul(amplitudes.T,basis) 
    
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        return self.dataset[idx]
        
    

    
