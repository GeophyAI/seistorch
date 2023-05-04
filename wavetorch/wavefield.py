class Wavefield:


    def __init__(self, equation="acoustic"):
        assert equation in ["acoustic", "elastic", "viscoacoustic"]
        self.wavefields = self.__getattribute__(equation)

    @property
    def acoustic(self,):
        return ["h1", "h2"]
    
    @property
    def elastic(self,):
        return ["vx", "vz", "txx", "tzz", "txz"]
    
    @property
    def viscoacoustic(self,):
        return ["vx", "vz", "p", "r"]