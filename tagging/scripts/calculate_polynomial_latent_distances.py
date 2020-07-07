import numpy as np
import sys
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from scipy import spatial
import pickle
import argparse
import pandas as pd


###########################################################

parser = argparse.ArgumentParser()
parser.add_argument("--data_file", type=str, default="train/spectra_noiseless.pd", help="file to used for training")
parser.add_argument("--n_pca", type=int, default=2, help="number of components used by pca")
parser.add_argument("--n_degree", type=int, default=4, help="degree of the polynomial")
parser.add_argument("--savepath", type=str, help="where to save the pickled files containing results")

opt = parser.parse_args()

###############################################################
n_degree = opt.n_degree
print("number of degrees of freedom:{}".format(n_degree))
print("pca componentsn:{}".format(opt.n_pca))

################################################

print("Loading data...")
#data_file = "spectra_noiseless"
#data = pd.read_pickle("/share/rcifdata/ddm/flatiron/taggingPaper/data/final/train/{}.pd".format(data_file))
#data = pd.read_pickle("/share/rcifdata/ddm/flatiron/taggingPaper/data/final/train/spectra_noiseless.pd")
data = pd.read_pickle(opt.data_file)
#data_noisy = pd.read_pickle(opt.data_file)

print("dataset is:{}".format(opt.data_file))


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
dataset_latents = pca.fit_transform(res_matrix)


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