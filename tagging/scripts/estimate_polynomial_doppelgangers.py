import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import sys
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from scipy import spatial
import pickle
import argparse


###########################################################

parser = argparse.ArgumentParser()
parser.add_argument("--data_file", type=str, default="/share/splinter/ddm/taggingClean/data/final/validation/spectra_noiseless.pd", help="file to used for training")
parser.add_argument("--savepath", type=str, help="where to save the pickled files containing results")
parser.add_argument("--n_pca", type=int, default=2, help="number of components used by pca")
parser.add_argument("--n_degree", type=int, default=4, help="degree of the polynomial")
opt = parser.parse_args()

###############################################################
n_degree = opt.n_degree
print("number of degrees of freedom:{}".format(n_degree))
print("pca components:{}".format(opt.n_pca))

################################################

print("Loading data...")
#data_file = "spectra_noiseless"
#data = pd.read_pickle("/share/rcifdata/ddm/flatiron/taggingPaper/data/final/train/{}.pd".format(data_file))
#data = pd.read_pickle("/share/rcifdata/ddm/flatiron/taggingPaper/data/final/train/spectra_noiseless.pd")
data = pd.read_pickle(opt.data_file)
#data_noisy = pd.read_pickle(opt.data_file)

print("dataset is:{}".format(opt.data_file))



#dataset = ApogeeDataset(data[:50000],n_bins)
####################################################

spectra_matrix = np.matrix(data["spectra"].tolist())
spectra_matrix = spectra_matrix[0:50000]

params_list = data.params.tolist()
params_list = params_list[0:50000]

polynomial = PolynomialFeatures(degree=opt.n_degree)
params_matrix = polynomial.fit_transform(np.array(params_list))
d = np.dot(np.linalg.inv(np.dot(params_matrix.T,params_matrix)),params_matrix.T)
s= np.dot(d,spectra_matrix)

fit_matrix = np.dot(params_matrix,s)
print(fit_matrix)
res_matrix = spectra_matrix - fit_matrix

pca = PCA(n_components=opt.n_pca)
latent_res = pca.fit_transform(res_matrix)

latent_ranking=[]
latent_tree = spatial.KDTree(latent_res)

no_siblings = 0
#need to apply PCA
for i in range(25000):
    d,idx2 = latent_tree.query(latent_res[i],p=1,k=50000)
    if 25000+i in idx2:
        pos = np.where(idx2==25000+i)[0][0]
        print("{} is:{}".format(i,pos))
        latent_ranking.append(pos)
        if pos == 1:
            no_siblings+=1
    else:
        print("{} is:{}".format(i,1000))
        latent_ranking.append(50000)

print("stars without siblings:{}".format(no_siblings))

with open(opt.savepath,"wb") as f:
    pickle.dump(latent_ranking,f)


