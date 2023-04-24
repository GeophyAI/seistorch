class Shape():
    def __init__(self, cfg, **kwargs):

        self.cfg = cfg
        equation = cfg['equation']
        # Set global values for shapes
        self.nt = cfg['geom']['nt']
        self.orinx = cfg['geom']['_oriNx']
        self.nshots = cfg['geom']['Nshots']

        self.channel = 2 if equation == 'elastic' else 1
        # Init library
        self.__init_lib__()
        # Build shape based on dictionary
        self.grad2d = self.grad2d_lib[equation]
        self.grad3d = self.grad3d_lib[equation]
        self.record3d = self.record3d_lib[equation]
        self.record2d = self.record2d_lib[equation]


    def __init_lib__(self,):

        self.record3d_lib = {"acoustic": self.__record3d_acoustic, 
                             "elastic" : self.__record3d_elastic}

        self.record2d_lib = {"acoustic": self.__record2d_acoustic, 
                             "elastic" : self.__record2d_elastic}
        
        self.grad2d_lib = {'acoustic': self.__grad2d_acoustic, 
                           'elastic': self.__grad2d_elastic}
        
        self.grad3d_lib = {'acoustic': self.__grad3d_acoustic, 
                           'elastic': self.__grad3d_elastic}
        
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
    def __grad2d_acoustic(self,):
        cfg = self.cfg
        return (1, cfg['geom']['Ny'], cfg['geom']['Nx'])

    @property
    def __grad3d_acoustic(self,):
        cfg = self.cfg
        return (self.nshots, cfg['geom']['Ny'], cfg['geom']['Nx'])
    
    @property
    def __grad2d_elastic(self,):
        cfg = self.cfg
        return (3, cfg['geom']['Ny'], cfg['geom']['Nx'])
        
    @property
    def __grad3d_elastic(self,):
        cfg = self.cfg
        return (self.nshots, 3, cfg['geom']['Ny'], cfg['geom']['Nx'])
    
    @property
    def __record2d(self,):
        cfg = self.cfg
        return (self.nt, self.orinx)

    @property
    def __record3d(self,):
        cfg = self.cfg
        return (self.nshots, self.nt, self.orinx)
    
    @property
    def __record3d_elastic(self,):
        cfg = self.cfg
        return (self.nshots, self.nt, self.orinx, self.channel)
    
    @property
    def __record3d_acoustic(self,):
        cfg = self.cfg
        return (self.nshots, self.nt, self.orinx, self.channel)

    @property
    def __record2d_elastic(self,):
        cfg = self.cfg
        return (self.nt, self.orinx, self.channel)
    
    @property
    def __record2d_acoustic(self,):
        cfg = self.cfg
        return (self.nt, self.orinx, self.channel)
