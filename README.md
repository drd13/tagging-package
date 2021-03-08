# Tagging

Companion repository containing code associated to  *Disentangled Representation Learning for Astronomical Chemical Tagging (de Mijolla, Ness, Viti, Wheeler 2021)*.
## Setup
### Dependencies
The code makes use of the following dependencies. Model training may not work with more recent pytorch releases.
* ```pip install torch torchvision sklearn```

### Downloading large files
As the trained models and datasets, required for running this codebase are too large to be shared through github, they must be downloaded externally. These can be found [here](https://drive.google.com/drive/u/1/folders/1pJvW3TMlgSByYqJtzRzgtIIlj62DoGFB).

The saved models (```factorDis.save```, ```faderDis.save```, ```faderDiswFe.save```) should be placed in ```tagging-package/outputs/models``` and the datasets (```spectra_noiseless.pd```,```spectra_noiseless_val.pd```) should be placed in ```tagging-package/data/processed```

To create the datasets in which noise is added, run the following commands in the terminal (assuming that terminal is located at ```tagging-package/```)
```
>>> python3 data/scripts/noiseSpectra.py --dataset_path=data/processed/spectra_noiseless.pd --new_name=spectra

>>> python3 data/scripts/noiseSpectra.py --dataset_path=data/processed/spectra_noiseless_val.pd --new_name=spectra_val
```

## Usage

Models can be trained from scratch by using the scripts ```/tagging-package/tagging/scripts/train_factordis.py``` and ```/tagging-package/tagging/scripts/train_faderdis.py```. (Be warned that models as included in the paper took roughly 24 hours to train on a GPU from an HPC cluster.)

Commented notebooks for reproducing the figures and analysing spectra can be found in ```tagging-package/tagging/notebooks```

