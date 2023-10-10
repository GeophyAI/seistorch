
class ConfigureCheck:

    def __init__(self, cfg):
        self.cfg = cfg

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
        
        assert isinstance(smooth["radius"], int), \
            f"keyword <radius> should be int."
        
        assert isinstance(smooth["sigma"], dict), \
            f"keyword <sigma> should be dict."

        for key in smooth["sigma"].keys():
            assert isinstance(smooth["sigma"][key], (int, float)), \
                f"sigma {key} should be float."

    