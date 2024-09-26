import h5py
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrand
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils import data
from jax.tree_util import tree_map
from seistorch.array import TensorList

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
                 MULTIPLE=True, 
                 Preload=True):

        self.dpath = dpath
        self.dkey = dkey
        self.srclist = srclist
        self.reclist = reclist
        self.padding = PMLN
        self.multiple = MULTIPLE

        assert len(self.srclist) == len(self), \
                "The number of sources and shots in data must be the same."
        if Preload:
            self.preload()

    def preload(self, ):
        nshots = len(self)
        self.data = []
        with h5py.File(self.dpath, 'r') as f:
            for i in range(nshots):
                self.data.append(f[f'{self.dkey}_{i}'][...])
        self.data = jnp.array(self.data)

    def __getitem__(self, key):
    
        src_this_epoch = self.srclist[key]
        rec_this_epoch = self.reclist[key]

        # Load data from h5df file
        with h5py.File(self.dpath, 'r') as f:
            if isinstance(key, list):
                data = TensorList([f[f'{self.dkey}_{k}'][...].copy() for k in key])
            if isinstance(key, int):
                data = f[f'{self.dkey}_{key}'][...].copy()
            else:
                data = self.data[key]
                
        return data, src_this_epoch, rec_this_epoch, key
            
    def __len__(self):
        # count the number of shots
        count = 0
        with h5py.File(self.dpath, 'r') as f:
            for dataset_name in f.keys():
                if dataset_name.startswith(self.dkey):
                    count += 1
        return count

def numpy_collate(batch):
    return tree_map(np.asarray, data.default_collate(batch))

class NumpyLoader(DataLoader):
    def __init__(self, dataset, batch_size=1,shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn)

class BaseDataLoader:
    """Dataloader Interface"""
    
    def __init__(
        self, 
        dataset, 
        batch_size: int = 1,  # batch size
        shuffle: bool = False,  # if true, dataloader shuffles before sampling each batch
        num_workers: int = 0,  # how many subprocesses to use for data loading.
        drop_last: bool = False,
        **kwargs
    ):
        pass

    def __len__(self):
        raise NotImplementedError
    
    def __next__(self):
        raise NotImplementedError
    
    def __iter__(self):
        raise NotImplementedError

def to_jax_dataset(dataset):
    if isinstance(dataset, OBSDataset):
        dataset.srclist = jnp.array(dataset.srclist)
        dataset.reclist = jnp.array(dataset.reclist)
        return dataset
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset)}")

class DataLoaderJAX(BaseDataLoader):

    def __init__(
        self, 
        dataset, 
        batch_size: int = 1,  # batch size
        shuffle: bool = False,  # if true, dataloader shuffles before sampling each batch
        num_workers: int = 0,  # how many subprocesses to use for data loading. Ignored.
        drop_last: bool = False,
        **kwargs
    ):
        self.key = jrand.PRNGKey(44)
        self.dataset = to_jax_dataset(dataset)
        
        self.indices = jnp.arange(len(dataset))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def reset_key(self, new_key):
        self.key = new_key
    
    def __iter__(self, key=None):
        indices = jrand.choice(self.key, self.indices, (self.batch_size,), replace=False)
        indices = indices.astype(int)
        batch = [self.dataset[idx] for idx in indices]
        
        yield [jnp.array(x) for x in zip(*batch)]
    
    def next_key(self):
        self.key, subkey = jrand.split(self.key)
        return subkey
    
    def __len__(self):
        complete_batches, remainder = divmod(len(self.indices), self.batch_size)
        return complete_batches if self.drop_last else complete_batches + bool(remainder)
