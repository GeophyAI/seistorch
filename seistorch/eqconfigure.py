class Parameters:
    """ 
    Specify which model parameters are required by a given equation. 
    """
    @staticmethod
    def valid_model_paras():

        paras = {"aec":          ["vp", "vs", "rho"], 
                 "acoustic":     ["vp"],
                 "elastic":      ["vp", "vs", "rho"],
                 "acoustic1st":  ["vp", "rho"],
                 #"ttielastic":   ["vp", "vs", "rho", "epsilon", "gamma", "delta"],
                 "ttielastic":   ["c11", "c13", "c33", "c15", "c35", "c55", "rho"],
                }

        return paras



class Wavefield:
    """ 
    Specify which wavefield variables are required by a given equation. 
    """
    def __init__(self, equation="acoustic"):
        self.wavefields = getattr(self, equation)

    @property
    def acoustic(self,):
        return ["h1", "h2"]
    
    @property
    def ttielastic(self,):
        return ["vx", "vz", "txx", "tzz", "txz"]
    
    @property
    def acoustic1st(self,):
        return ["vx", "vz", "p"]
    
    @property
    def elastic(self,):
        return ["vx", "vz", "txx", "tzz", "txz"]
    
    @property
    def viscoacoustic(self,):
        return ["vx", "vz", "p", "r"]
    
    @property
    def aec(self,):
        return ["p", "vx", "vz", "txx", "tzz", "txz"]

class Shape():
    def __init__(self, cfg, **kwargs):

        self.cfg = cfg
        self.length_invert = len([invert for invert in cfg['geom']['invlist'].values() if invert == 1])
        self.length_invert = max(self.length_invert, 1)
        equation = cfg['equation']
        # Set global values for shapes
        self.nt = cfg['geom']['nt']
        self.orinx = cfg['geom']['_oriNx']
        self.nshots = cfg['geom']['Nshots']

        # Init library
        # Build shape based on dictionary
        self.grad2d = self.__grad2d__# self.grad2d_lib[equation]
        self.grad3d = self.__grad3d__# self.grad3d_lib[equation]
        
    @property
    def numel(self,):
        cfg = self.cfg
        return cfg['geom']['Ny']* cfg['geom']['Nx']
    
    def model2d(self,):
        cfg = self.cfg
        return (cfg['geom']['Ny'], cfg['geom']['Nx'])
        
    @property
    def loss(self,):
        cfg = self.cfg
        return (len(cfg['geom']['multiscale']), cfg['training']['N_epochs'], self.nshots)
    
    @property
    def hessian(self,):
        cfg = self.cfg
        length = cfg['geom']['Ny']*cfg['geom']['Nx']
        return (length, length)

    @property
    def __grad2d__(self,):
        cfg = self.cfg
        return (self.length_invert, cfg['geom']['Ny'], cfg['geom']['Nx'])

    @property
    def __grad3d__(self,):
        cfg = self.cfg
        return (self.nshots, self.length_invert, cfg['geom']['Ny'], cfg['geom']['Nx'])
    
    @property
    def __record2d__(self,):
        cfg = self.cfg
        return (self.nt, self.orinx, self.channels)

    @property
    def __record3d__(self,):
        cfg = self.cfg
        return (self.nshots, self.nt, self.orinx, self.channels)
