import pandas as pd
import numpy as np
import pickle

with open("/mnt/home/dmijolla/taggingPaper/data/external/s_train_merged","rb") as f:       
    data = pickle.load(f)

a = data
filtered_a = []
for i in range(len(a)):                      
   print(i)
   filtered_a.append({"params":[a[i]["t_eff"],a[i]["log_g"],a[i]["metals"],a[i]["am"],a[i]["cm"]],"spectra":np.nan_to_num(a[i]["spectra"]),"abundances":np.array(a[i]["abundances"])[:,1]})
   data_processed = pd.DataFrame(filtered_a)

print("saving")
data_processed.to_pickle("/mnt/home/dmijolla/taggingPaper/data/final/train/s_train_merged.pd")
print("finished saving")
