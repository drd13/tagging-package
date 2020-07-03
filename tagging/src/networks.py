import torch.nn as nn
import torch
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ConditioningAutoencoder(nn.Module):
    def __init__(self,encoder,decoder,n_bins = None,n_embedding=None):
        super(ConditioningAutoencoder, self).__init__()
        self.n_bins = n_bins
        self.w = nn.Embedding(self.n_bins, n_embedding)
        self.encoder = encoder
        self.decoder = decoder

        
    def forward(self, x,u,t=None,train_encoder=True,train_decoder=True):
        #IDEA:we train a neural network to reconstruct the continuum from only T and logg and then add this.
        #u is the variable we are marginalizing over
        latent=None
        output=None
        if train_encoder:
            x = torch.cat((x,u.float()),1)
            x= self.encoder(x)
            latent = x
        if train_decoder:
            x = torch.cat((x,u),1)
            x = self.decoder(x,t,self.w)
            output = x
        return output,latent

class ConditioningNosiyAutoencoder(nn.Module):
    def __init__(self,encoder,decoder,noise=0,n_bins = None,n_embedding=None):
        super(ConditioningAutoencoder, self).__init__()
        self.n_bins = n_bins
        self.w = nn.Embedding(self.n_bins, n_embedding)
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, x,u,t=None,train_encoder=True,train_decoder=True):
        #IDEA:we train a neural network to reconstruct the continuum from only T and logg and then add this.
        #u is the variable we are marginalizing over
        latent=None
        output=None
        if train_encoder:
            x = torch.cat((x,u.float()),1)
            x= self.encoder(x)
            latent = x
        if train_decoder:
            noise = x.data.new(x.size()).normal_(0.,1/noise)*4
            x = torch.cat((x,u),1)
            x = self.decoder(x,t,self.w)
            output = x
        return output,latent



class ConditioningHypersphereAutoencoder(nn.Module):
    def __init__(self,encoder,decoder,n_bins = None,n_embedding=None):
        super(ConditioningHypersphereAutoencoder, self).__init__()
        self.n_bins = n_bins
        self.encoder = encoder
        self.decoder = decoder

        
    def forward(self, x,u,t=None,train_encoder=True,train_decoder=True):
        #IDEA:we train a neural network to reconstruct the continuum from only T and logg and then add this.
        #u is the variable we are marginalizing over
        latent=None
        output=None
        if train_encoder:
            x = torch.cat((x,u.float()),1)
            x= self.encoder(x)
            x_norm = x.norm(p=2, dim=1, keepdim=True)
            x = x.div(x_norm)
            latent = x
        if train_decoder:
            x = torch.cat((x,u),1)
            x = self.decoder(x,t,None)
            output = x
        return output,latent




class ConditioningVAE(nn.Module):
    def __init__(self,encoder,decoder,n_bins = None):
        super(ConditioningVAE, self).__init__()
        self.n_bins = n_bins
        self.encoder = encoder
        self.decoder = decoder


    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, x,u,t=None,train_encoder=True,train_decoder=True):
        #IDEA:we train a neural network to reconstruct the continuum from only T and logg and then add this.
        #u is the variable we are marginalizing over
        latent=None
        output=None
        if train_encoder:
            x = torch.cat((x,u.float()),1)
            mu,logvar= self.encoder(x)
            z = self.reparametrize(mu,logvar)
        if train_decoder:
            z = torch.cat((z,u),1)
            x = self.decoder(z,None,None)
            output = x
        return output,mu,logvar

class EncoderVAE(nn.Module):
    def __init__(self,encoder,branch_1,branch_2):
        super(EncoderVAE, self).__init__()
        self.encoder =encoder
        self.branch_1 = branch_1
        self.branch_2 = branch_2
        
    def forward(self, x):
        z = self.encoder(x)
        mu = self.branch_1(z)
        logvar = self.branch_2(z)
        return mu,logvar



class Embedding_Decoder(nn.Module):
    def __init__(self,decoder,pre_decoder=None,n_batch = 64,n_bins=None):
        super(Embedding_Decoder, self).__init__()
        self.decoder = decoder
        self.pre_decoder = pre_decoder
        self.n_used = 256
        self.n_batch = n_batch
        self.n_bins = n_bins
        
    def forward(self,x,t,w, n_reconstructed = None):
        #For training to proceed smoothly, we only reconstruct a subset of all inputs 
        #if not n_reconstructed:
        #  n_reconstructed = 
        x = torch.unsqueeze(x,1)
        if self.pre_decoder:
          x = pre_decoder(x)
        if t is None:
          t = torch.cuda.LongTensor(range(self.n_bins))
        x =x.repeat(1,len(t),1)
        w_val = w(t)
        w_val = torch.unsqueeze(w_val,0)
        w_val =w_val.repeat(self.n_batch,1,1)
        x_input = torch.cat((w_val,x.float()),2)
        output= self.decoder(x_input)
        output = torch.squeeze(output,2)
        return output
      
      
      

class Feedforward(nn.Module):
    def __init__(self,structure,activation=nn.LeakyReLU(),with_dropout=False):
        super(Feedforward, self).__init__()        
        self.layers = []
        for i in range(len(structure)-2):
            if i==1 and with_dropout==True:
              print("using dropout")
              self.layers.append(nn.Dropout(0.5))
            self.layers.append(nn.Linear(structure[i],structure[i+1]))
            self.layers.append(activation)
        
        self.layers.append(nn.Linear(structure[-2],structure[-1]))
        self.fc = nn.Sequential(*self.layers)


    def forward(self, x,optional=False,optional2=False):
        #optional was added so that feedforward took as many inputs as embedding_decoder
        #for layer in layers:
        #    x = layer(x)
        x = self.fc(x)
        return x



class FeedforwardDropout(nn.Module):
    def __init__(self,structure,activation=nn.LeakyReLU(),with_dropout=False,p=0.1):
        super(FeedforwardDropout, self).__init__()        
        self.layers = []
        self.p = p
        for i in range(len(structure)-2):
            self.layers.append(nn.Linear(structure[i],structure[i+1]))
            self.layers.append(activation)
            self.layers.append(nn.Dropout(self.p))
        
        self.layers.append(nn.Linear(structure[-2],structure[-1]))
        self.fc = nn.Sequential(*self.layers)


    def forward(self, x,optional=False,optional2=False):
        #optional was added so that feedforward took as many inputs as embedding_decoder
        #for layer in layers:
        #    x = layer(x)
        x = self.fc(x)
        return x

class FeedforwardBatchnorm(nn.Module):
    def __init__(self,structure,activation=nn.LeakyReLU(),with_dropout=False,p=0.1):
        super(FeedforwardBatchnorm, self).__init__()        
        self.layers = []
        self.p = p
        for i in range(len(structure)-2):
            self.layers.append(nn.Linear(structure[i],structure[i+1]))
            self.layers.append(activation)
            self.layers.append(nn.BatchNorm1d(structure[i+1]))
        
        self.layers.append(nn.Linear(structure[-2],structure[-1]))
        self.fc = nn.Sequential(*self.layers)


    def forward(self, x,optional=False,optional2=False):
        #optional was added so that feedforward took as many inputs as embedding_decoder
        #for layer in layers:
        #    x = layer(x)
        x = self.fc(x)
        return x





class Matching_Network(nn.Module):
    def __init__(self, network):
        #network is shown one batch with matching marginalized parameters and one batch with shuffled parameters.
        #The network assigns a probability of matching to each pair
        #The matching network is interested in maximally seperating the two cases. That is to say minimize err(p(shuffled),0)+err(p(not_shuffled),1)
        #The Conditional_Autoencoder is interested in fooling the matching network that is to say to maximize that quantity
        super().__init__()
        self.network = network
        self.sigmoid = nn.Sigmoid()

    def forward(self, z, u):
        z = torch.cat((z, u), dim=1)
        x = self.network(z)
        p = self.sigmoid(x)
        return p
      


class Discriminator(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        x = self.network(z)
        p = self.sigmoid(x)
        return p
      


