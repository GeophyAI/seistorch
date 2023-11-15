import torch

class SeisCompile:

    def __init__(self, logger=None):
        self.logger = logger

    def compile(self, func, **kwargs):

        if self.logger is not None:
            self.logger.print(f"Trying to compile function <{func.__name__}>...")

        compile_ok = self.torch_version_ok() and self.gpu_ok()

        if not compile_ok:
            if self.logger is not None:
                self.logger.print(f"Compile is not supported in this environment.")
            return func
        
        if self.logger is not None:
            self.logger.print(f"Compiling function <{func.__name__}> successfully")

        return torch.compile(func, **kwargs)
            

    def gpu_ok(self,):

        gpu_ok = False

        if torch.cuda.is_available():
            device_cap = torch.cuda.get_device_capability()
            if device_cap in ((7, 0), (8, 0), (9, 0)):
                gpu_ok = True

        if not gpu_ok:
            if self.logger is not None:
                self.logger.print(
                    "GPU is not NVIDIA V100, A100, or H100."
                )
            
        return gpu_ok

    def torch_version_ok(self,):

        main_torch_version = int(torch.__version__.split('.')[0])
        version_ok = True if main_torch_version > 1 else False

        if not version_ok:
            if self.logger is not None:
                self.logger.print(
                    "If you want to use compile, please upgrade your torch to 2.0 or higher."
                )

        return version_ok