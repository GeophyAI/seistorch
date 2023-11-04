import os
import logging

class SeisLog(logging.Logger):

    def __init__(self, name="Seistorch", level=logging.DEBUG, backend="MPI"):
        super(SeisLog, self).__init__(name, level)

        assert backend in ["MPI", "LOCAL", "TORCHRUN"], "Only support MPI and LOCAL backend."

        backend_init = getattr(self, backend.lower()+"_backend")
        backend_init()

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(level)
        self.console_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.console_handler)

    def mpi_backend(self):

        from mpi4py import MPI
        from mpi4py.util import pkl5

        comm = pkl5.Intracomm(MPI.COMM_WORLD)
        self.rank = comm.Get_rank()

    def local_backend(self):

        self.rank = 0

    def torchrun_backend(self):
            
        self.rank = os.getenv('LOCAL_RANK')
        assert self.rank is not None, "Please use the script with torchrun."
        self.rank = int(self.rank)

    def print(self, msg):
        if self.rank==0: 
            self.logger.info(msg) 

