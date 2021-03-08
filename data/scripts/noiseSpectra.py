"""
Cleans spectras (if required) and adds equivalent noisy version of datasets
"""

import pandas as pd
import pickle
import numpy as np
import click
import os

@click.command()
@click.option('--dataset_path',default="../processed/spectra_noiseless.pd", help='path to dataset')
@click.option('--new_name',default="spectra", help='name of new dataset paths')
def make_dataset(dataset_path,new_name,SNs = [10,30,50,100]):
    df = pd.read_pickle(dataset_path)
    df["spectra"] = df["spectra"].apply(remove_zeros)
    #df.to_pickle(dataset_path)
    folder_path = os.path.split(dataset_path)[0]
    for SN in SNs:
        df_new =df.copy()
        df_new["spectra"] = df_new["spectra"].apply(add_noise,noise=1/SN)
        df_new.to_pickle("{}/{}_SN_{}.pd".format(folder_path,new_name,SN))


def remove_zeros(spectra): 
    return  spectra[np.nonzero(spectra)[0]]

def add_noise(spectra,noise): #need to add some noise to the spectra
    noise = np.random.normal(0,noise,len(spectra))
    return noise+spectra


if __name__ == '__main__':
    make_dataset()

