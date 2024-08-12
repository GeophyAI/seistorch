import os
from .eqconfigure import Parameters, Wavefield

class ConfigureCheck:

    def __init__(self, cfg, mode="forward", args=None):
        self.cfg = cfg
        self.mode = mode
        self.inverion = self.mode == "inversion"
        self.forward = self.mode == "forward"
        self.args = args
        self.__setup__()

    def __setup__(self,):

        self.check_equation()
        self.check_source_receiver_type()
        self.check_path()
        self.check_boundary()
        
        if self.args is not None:
            if self.args.grad_smooth:
                self.check_smooth()
            if self.args.grad_cut:
                self.check_seabed()

    def check_boundary(self, title='boundary'):
        """Check the boundary parameters.
        """
        assert title in self.cfg["geom"].keys(), \
            f"boundary parameter is not in the config file."
        
        boundary = self.cfg["geom"][title]
        assert isinstance(boundary, dict), \
            f"boundary should be dict with keys <type, width>."
        
        assert "type" in boundary.keys(), \
            f"type is not in the boundary parameter."
        
        assert boundary["type"] in ["pml", "habc", "random"], \
            f"boundary type should be 'pml' or 'habc'."
        
        assert "width" in boundary.keys(), \
            f"width is not in the boundary parameter."
        
        assert boundary['width'] == 50, \
            f"Currently, the width of the boundary should be 50."
        
        if boundary["type"] == 'habc':
            assert 'habc' in self.cfg['equation'], \
                f'When boundary type is habc, the equation must be <...>_habc.'

    def check_dict(self, key, dict):
        assert key in dict.keys(), \
            f"{key} is not in the config file."

    def check_equation(self, title="equation"):

        modelPath = self.cfg['VEL_PATH']
        invlist = self.cfg['geom']['invlist']
        needed_model_paras = Parameters.valid_model_paras()[self.cfg['equation']]

        for para in needed_model_paras:
            # check if the model of <para> is in the modelPath
            if para not in modelPath.keys():
                print(f"Model '{para}' is not found in modelPath")
                exit()
            # check if the model of <para> is in the invlist
            if para not in invlist.keys():
                print(f"Model '{para}' is not found in invlist")
                exit()
            # check the existence of the model file
            if not os.path.exists(modelPath[para]):
                print(f"Cannot find model file '{modelPath[para]}' which is needed by equation <{self.cfg['equation']}>")
                exit()

    def check_path(self, ):
        """Check the path of the config file.
        """
        geoms = ['sources', 'receivers']
        for geom in geoms:
            assert os.path.exists(self.cfg['geom'][geom]), \
                f"Cannot find {geom} file '{self.cfg['geom'][geom]}'"

    def check_smooth(self, 
                     title="smooth", 
                     keys=["counts", "radius", "sigma"]):
        """Check the smooth parameters.
        """

        # check the existence of the keyword
        assert title in self.cfg["training"].keys(), \
            f"smooth parameter is not in the config file."
        
        smooth = self.cfg["training"][title]

        # check the existence of the keys
        for key in keys:
            assert key in smooth.keys(), \
                f"{key} is not in the smooth parameter."
            
        # check the type of the keys
        assert isinstance(smooth["counts"], int), \
            f"keyword <counts> should be int."
        
        assert isinstance(smooth["radius"], dict), \
            f"keyword <radius> should be dict with keys <x, y, z>."
        
        for key in smooth["radius"].keys():
            assert isinstance(smooth["radius"][key], (int, float)), \
                f"radius {key} should be float."
        
        assert isinstance(smooth["sigma"], dict), \
            f"keyword <sigma> should be dict with keys <x, y, z>."

        for key in smooth["sigma"].keys():
            assert isinstance(smooth["sigma"][key], (int, float)), \
                f"sigma {key} should be float."
     
    def check_source_receiver_type(self,):
        
        wavefield_names = Wavefield(self.cfg['equation']).wavefields

        # Check source type:
        for source_type in self.cfg['geom']['source_type']:
            assert source_type in wavefield_names, \
                f"Valid source type are {wavefield_names}, but got '{source_type}'. Please check your configure file."

        # Check receiver type:
        for recev_type in self.cfg['geom']['receiver_type']:
            if 'lsrtm' in self.cfg['equation']:
                assert recev_type.startswith('s'), \
                    f"Receiver type should start with 's' in lsrtm equations."
            assert recev_type in wavefield_names, \
                f"Valid receiver type are {wavefield_names}, but got '{recev_type}'. Please check your configure file."

    def check_seabed(self, title="seabed"):
        """Check the seabed parameters.
        """
        assert title in self.cfg["geom"].keys(), \
            f"When you specify <--grad-cut> in commands, you must also specify <geom>-><seabed> in config file."