import os
import logging

class SeisLog(logging.Logger):

    def __init__(self, name, level=logging.DEBUG):
        super(SeisLog, self).__init__(name, level)
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(level)
        self.console_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.console_handler)

    def print(self, msg):
        # rank = os.getenv('LOCAL_RANK')
        # assert rank is not None, "Please use the script with torchrun."
        # if rank==0: 
        self.logger.info(msg) 
