import torch
from .utils import to_tensor
from .utils import diff_using_roll
from .checkpoint import checkpoint as ckpt

    
def _time_step(vp, vs, rho, p, vx, vz, txx, tzz, txz, dt, h, d):
    lame_lambda = rho*(vp.pow(2)-2*vs.pow(2))
    lame_mu = rho*(vs.pow(2))

    vx_x = diff_using_roll(vx, 2, False)
    vz_z = diff_using_roll(vz, 1)
    vx_z = diff_using_roll(vx, 1)
    vz_x = diff_using_roll(vz, 2)

    c = 0.5*dt*d

    y_txx = (1+c)**-1*(dt*lame_mu*h.pow(-1)*(vx_x-vz_z)+(1-c)*txx)

    y_tzz = (1+c)**-1*(dt*lame_mu*h.pow(-1)*(vz_z-vx_x)+(1-c)*tzz)

    y_txz = (1+c)**-1*(dt*lame_mu*h.pow(-1)*(vz_x+vx_z)+(1-c)*txz)

    txx_x = diff_using_roll(y_txx, 2)
    txz_z = diff_using_roll(y_txz, 1, False)
    tzz_z = diff_using_roll(y_tzz, 1, False)
    txz_x = diff_using_roll(y_txz, 2, False)

    y_p = (1+c)**-1*(-dt*(lame_lambda+lame_mu)*h.pow(-1)*(vx_x+vz_z)+(1-c)*p)

    p_x = diff_using_roll(y_p, 2)
    p_z = diff_using_roll(y_p, 1, False)

    # Update y_vx
    y_vx = (1+c)**-1*(dt*rho.pow(-1)*h.pow(-1)*(txx_x+txz_z-p_x)+(1-c)*vx)
    # Update y_vz
    y_vz = (1+c)**-1*(dt*rho.pow(-1)*h.pow(-1)*(txz_x+tzz_z-p_z)+(1-c)*vz)

    return y_p, y_vx, y_vz, y_txx, y_tzz, y_txz


class WaveCell(torch.nn.Module):
    """The recurrent neural network cell implementing the scalar wave equation"""

    def __init__(self, geometry):

        super().__init__()

        # Set values
        self.geom = geometry
        self.register_buffer("dt", to_tensor(self.geom.dt))

    def parameters(self, recursive=True):
        for param in self.geom.parameters():
            yield param

    def get_parameters(self, key=None, recursive=True):
        yield self.geom.__getattr__(key)

    def forward(self, wavefields, model_vars, **kwargs):
        """Take a step through time
        Parameters
        ----------
        vx, vz, txx, tzz, txz : 
             wave field one time step ago
        vp  :
            Vp velocity.
        vs  :
            Vs velocity.
        rho : 
            Projected density, required for nonlinear response (this gets passed in to avoid generating it on each time step, saving memory for backprop)
        """
        t = kwargs['t']
        it = kwargs['it']
        p, vx, vz, txx, tzz, txz = wavefields
        vp, vs, rho = model_vars

        checkpoint = self.geom.checkpoint
        forward = not self.geom.inversion
        inversion = self.geom.inversion

        if checkpoint and inversion:
            hidden = ckpt(_time_step, t%it==0, vp, vs, rho, p, vx, vz, txx, tzz, txz, self.dt, self.geom.h, self.geom.d)
        if forward or (inversion and not checkpoint):
            hidden = _time_step(vp, vs, rho, p, vx, vz, txx, tzz, txz, self.dt, self.geom.h, self.geom.d)

        return hidden