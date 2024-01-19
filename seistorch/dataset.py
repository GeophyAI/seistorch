import h5py
from torch.utils.data import Dataset
from seistorch.type import TensorList
import numpy as np
from seistorch.setup import setup_src_coords, setup_rec_coords

class OBSDataset(Dataset):
    """The class for data loading.
       Supported file types: .npy, .hdf5
       For numpy files, we load the whole data into memory.
       For hdf5 files, we can load a single shot into memory.
    """

    def __init__(self, 
                 dpath, 
                 dkey='shot', 
                 srclist=None, 
                 reclist=None, 
                 freqs=None, 
                 PMLN=50, 
                 MULTIPLE=True):
        self.dpath = dpath
        self.dkey = dkey
        self.srclist = srclist
        self.reclist = reclist
        self.padding = PMLN
        self.multiple = MULTIPLE

        assert len(self.srclist) == len(self), \
                "The number of sources and shots in data must be the same."
            

    def __getitem__(self, key):

        src_this_epoch = self.srclist[key]
        rec_this_epoch = self.reclist[key]

        # Load data from h5df file
        with h5py.File(self.dpath, 'r') as f:
            if isinstance(key, list):
                data = TensorList([f[f'{self.dkey}_{k}'][...].copy() for k in key])
            if isinstance(key, int):
                data = f[f'{self.dkey}_{key}'][...].copy()
        return data, src_this_epoch, rec_this_epoch, key
            
    def __len__(self):
        # count the number of shots
        count = 0
        with h5py.File(self.dpath, 'r') as f:
            for dataset_name in f.keys():
                if dataset_name.startswith(self.dkey):
                    count += 1
        return count