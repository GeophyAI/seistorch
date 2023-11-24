
import os
import pickle
import torch
import numpy as np
from .type import TensorList
from obspy import Stream, Trace
from yaml import load, dump
from yaml import CLoader as Loader
import h5py


class SeisRecord:

    def __init__(self, cfg, logger=None):

        self.cfg = cfg
        self.logger = logger
        self.io = SeisIO(load_cfg=False)
        
        self.datapath = cfg['geom']['obsPath']
        self.recpath = cfg['geom']['receivers']

        self.nshots = len(self.io.read_pkl(self.recpath))

        self.filetype = self.io.get_file_extension(self.datapath)

    def create_hdf5_file(self, datapath=None, recpath=None):

        if datapath is None:
            datapath = self.datapath

        if recpath is None:
            recpath = self.recpath

        rec_groups = self.load_receivers(recpath)
        nshots = len(rec_groups)
        nc = len(self.cfg['geom']['receiver_type'])
        nt = self.cfg['geom']['nt']

        with h5py.File(datapath, 'w') as f:
            for shot in range(nshots):
                nr = len(rec_groups[shot][0])
                # self.logger.print(f"Create shot {shot} with shape ({nt}, {nr}, {nc}).")
                f.create_dataset(f'shot_{shot}', (nt, nr, nc), dtype='f', chunks=True)
            print(f"Create file {datapath} successfully.")

    def load_receivers(self, path=None):

        if path is None:
            path = self.recpath

        return self.io.read_pkl(path)

    def setup(self, mode):

        if self.logger is not None:
            self.logger.print(f"Detect filetype is {self.filetype}.")

        _setup_ = getattr(self, f"setup_{mode}")
        _setup_()

    def setup_forward(self, ):
        if self.filetype == '.npy':
            # self.logger.print(f"Create empty numpy array with shape ({self.nshots}, ).")
            self.record = np.empty(self.nshots, dtype=np.ndarray)

        if self.filetype == '.hdf5':
            # self.logger.print("Create hdf5 file on disk.")
            self.create_hdf5_file()

    def setup_inversion(self, ):
        if self.filetype == '.npy':
            # self.logger.print(f"Load numpy array from {self.datapath}.")
            self.record = np.load(self.datapath, allow_pickle=True)

        if self.filetype == '.hdf5':
            pass
            # self.logger.print("Memory map hdf5 file on disk.")

    def write_shot(self, shot_no, data, datapath=None):
            
        if datapath is None:
            datapath = self.datapath

        if self.filetype == '.hdf5':
            with h5py.File(datapath, 'a') as f:
                f[f'shot_{shot_no}'][...] = data

        if self.filetype == '.npy':
            self.record[shot_no] = data
            np.save(datapath, self.record)

    @property
    def shape(self, ):
        if self.filetype == '.npy':
            return (self.record.shape[0],)

        if self.filetype == '.hdf5':
            with h5py.File(self.datapath, 'r') as f:
                return (len(f.keys()),)

    def __setitem__(self, key, value):
        if isinstance(key, int) and isinstance(value, np.ndarray):
            self.write_shot(key, value, self.datapath)
        else:
            raise NotImplementedError("The key must be an integer and the value must be a numpy array.")

    def __getitem__(self, key):

        if self.filetype == '.hdf5':
            return self.getitem_from_hdf5(key)
            # with h5py.File(self.datapath, 'r') as f:
            #     return f[f'shot_{key}'][...].copy()
            
        if self.filetype == '.npy':
            return self.record[key]

    def getitem_from_hdf5(self, key):
        with h5py.File(self.datapath, 'r') as f:
            if isinstance(key, list):
                return TensorList([f[f'shot_{k}'][...].copy() for k in key])
            if isinstance(key, int):
                return f[f'shot_{key}'][...].copy()

class SeisIO:
    """The class for file/data input and output.
    """

    def __init__(self, cfg: dict = None, cfg_path: str=None, load_cfg=False):
        """Initialize the SeisIO class."""
        if cfg is not None:
            self.cfg = cfg
        if load_cfg and cfg_path is not None:
            self.cfg = self.read_cfg(cfg_path)

    def fromfile(self, path: str=None):
        """Read the data.

        Args:
            path (str): The path to the data.
            dtype (str): The data type.

        Returns:
            np.ndarray: The data.
        """
        assert self.path_exists(path), f"Cannot found {path}."

        data_loader = self.decide_loader(path)

        return data_loader(path, allow_pickle=True)
    
    def decide_loader(self, path: str=None):
        """Decide the data loader.

        Args:
            path (str): The path to the data.

        Returns:
            function: The data loader.
        """

        extension = self.get_file_extension(path)

        data_loader = {".npy": np.load, 
                       ".bin": self.read_fortran_binary, 
                       ".pkl": self.read_pkl}

        if extension in data_loader:
            return data_loader[extension]
        else:
            raise NotImplementedError(f"Cannot read {extension} file.")

    def get_file_extension(self, path: str):
        """Get the extension of the file.

        Args:
            path (str): The path to the file.

        Returns:
            str: The extension of the file.
        """
        return os.path.splitext(path)[1]

    def gll2grid(self, data, x, z, grid_size:float):
        """Interpolate the data onto a regular grid.

        Args:
            data (gll data): The data to be interpolated.
            x (np.ndarray): The x coordinates of the data.
            z (np.ndarray): The z coordinates of the data.
            grid_size (float): The target grid size.

        Returns:
            np.ndarray: The interpolated data.
        """

        # calculate the grid sizes
        x_min, x_max = x.min(), x.max()
        z_min, z_max = z.min(), z.max()

        # Generate a grid of coordinates
        x_grid = np.arange(x_min, x_max + grid_size, grid_size)
        z_grid = np.arange(z_min, z_max + grid_size, grid_size)

        # Calculate the grid coordinates of the data
        x_indices = ((x - x_min) / grid_size).astype(int)
        z_indices = ((z - z_min) / grid_size).astype(int)

        # Interpolate the data onto a regular grid
        griddata = np.zeros((len(z_grid), len(x_grid)))
        griddata[z_indices, x_indices] = data

        return griddata

    def np2st(self, data, dt):
        """Convert the numpy array to the obspy stream.

        Args:
            data (np.ndarray): The data to be converted.
            dt (float): The sampling interval.

        Returns:
            stream: The obspy stream.
        """
        starttime = 0
        nt, nr = data.shape[:2]
        st = Stream()
        sampling_rate=1/dt
        # Loop over the traces
        for i in range(nr):
            trace_data = data[:, i]  # Get the data
            trace = Trace(data=trace_data, header={"starttime": starttime, "sampling_rate": sampling_rate})
            st += trace
        return st

    def np2tensor(self, data, device, dtype=None):
        return torch.from_numpy(data).to(device=device, dtype=dtype)

    def path_exists(self, path):
        """Check if the path exists.

        Args:
            path (str): The path to be checked.

        Returns:
            bool: True if the path exists, False otherwise.
        """
        path = '' if path is None else path

        return os.path.exists(path)

    def read_cfg(self, cfg_path):
        """Read the configure file.

        Args:
            cfg_path (str): The path to the configure file.

        Raises:
            FileNotFoundError: If the configure file is not found.

        Returns:
            cfg: The converted configure dictionary.
        """
        if self.path_exists(cfg_path):
            with open(cfg_path, 'r') as ymlfile:
                cfg = load(ymlfile, Loader=Loader)
            return cfg
        else:
            raise FileNotFoundError(f"Config file {cfg_path} not found.")

    def read_geom(self, cfg: dict=None):

        assert cfg is not None, "The configure file is not loaded."

        spath = cfg["geom"]["sources"]
        rpath = cfg["geom"]["receivers"]
        assert os.path.exists(spath), "Cannot found source file."
        assert os.path.exists(rpath), "Cannot found receiver file."
        source_locs = self.read_pkl(spath)
        recev_locs = self.read_pkl(rpath)
        # assert len(source_locs)==len(recev_locs), \
        #     "The lenght of sources and recev_locs must be equal."
        return source_locs, recev_locs

    def read_hdf5(self, path: str, shot_no: int=None, **kwargs):
        with h5py.File(path, 'r') as f:
            d = f[f'shot_{shot_no}'][...].copy()
        return d

    def read_pkl(self, path: str, **kwargs):
        """Read a pickle file.

        Args:
            path (str): The path to the pickle file.

        Returns:
            data: The data loaded from the pickle file.
        """
        # Open the file in binary mode and load the list using pickle

        self.path_exists(path)
        with open(path, 'rb') as f:
            data = pickle.load(f)

        return data

    def read_vel(self, path: str, pmln=50, expand=0):
        """Read the velocity model.

        Args:
            path (str): The path to the velocity model.
            pmln (int, optional): The width of PML. Defaults to 50.
            expand (int, optional): The widht of expandsion. Defaults to 0.
        """
        self.path_exists(path)

        extension = self.get_file_extension(path)

        vel_loader = {".npy": np.load, 
                      ".bin": self.read_fortran_binary}
        
        if extension in vel_loader:
            vel = vel_loader[extension](path)

        if pmln > 0:
            vel = vel[pmln:-pmln, pmln:-pmln]
        if expand > 0:
            vel = vel[:, expand:-expand]

        return vel

    def read_fortran_binary(self, filename, **kwargs):
        """
        Reads Fortran-style unformatted binary data into numpy array.

        .. note::
            The FORTRAN runtime system embeds the record boundaries in the data by
            inserting an INTEGER*4 byte count at the beginning and end of each
            unformatted sequential record during an unformatted sequential WRITE.
            see: https://docs.oracle.com/cd/E19957-01/805-4939/6j4m0vnc4/index.html

        :type filename: str
        :param filename: full path to the Fortran unformatted binary file to read
        :rtype: np.array
        :return: numpy array with data with data read in as type Float32
        """
        nbytes = os.path.getsize(filename)
        with open(filename, "rb") as file:
            # read size of record
            file.seek(0)
            n = np.fromfile(file, dtype="int32", count=1)[0]

            if n == nbytes - 8:
                file.seek(4)
                data = np.fromfile(file, dtype="float32")
                return data[:-1]
            else:
                file.seek(0)
                data = np.fromfile(file, dtype="float32")
                return data

    def to_file(self, path: str=None, data: np.ndarray=None):
        """Write the data to a file.

        Args:
            path (str): The path to the file.
            data (np.ndarray): The data to be written.
        """

        extension = self.get_file_extension(path)

        wavelet_writer = {".npy": np.save, 
                          ".bin": self.write_fortran_binary}

        if extension in wavelet_writer:
            wavelet_writer[extension](path, data)

    def wavelet_fromfile(self, path: str=None):
        """Read the wavelet.

        Args:
            path (str): The path to the wavelet.

        Returns:
            np.ndarray: The wavelet.
        """
        path = self.cfg['geom']['wavelet'] if path is None else path

        self.path_exists(path)

        extension = self.get_file_extension(path)

        wavelet_loader = {".npy": np.load, 
                          ".bin": self.read_fortran_binary}

        if extension in wavelet_loader:
            wavelet = wavelet_loader[extension](path)

        return wavelet

    def write_cfg(self, path: str, cfg: dict, ):
        """Write the configure file.

        Args:
            cfg (dict): The configure dictionary.
            path (str): The path to the configure file.
        """
        with open(path, 'w') as f:
            dump(cfg, f)

    def write_pkl(self, path: str, data: list):
        # Open the file in binary mode and write the list using pickle
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def write_fortran_binary(self, filename, arr):
        """
        Writes Fortran style binary files. Data are written as single precision
        floating point numbers.

        .. note::
            FORTRAN unformatted binaries are bounded by an INT*4 byte count. This
            function mimics that behavior by tacking on the boundary data.
            https://docs.oracle.com/cd/E19957-01/805-4939/6j4m0vnc4/index.html

        :type arr: np.array
        :param arr: data array to write as Fortran binary
        :type filename: str
        :param filename: full path to file that should be written in format
            unformatted Fortran binary
        """
        buffer = np.array([4 * len(arr)], dtype="int32")
        data = np.array(arr, dtype="float32")

        with open(filename, "wb") as file:
            buffer.tofile(file)
            data.tofile(file)
            buffer.tofile(file)
