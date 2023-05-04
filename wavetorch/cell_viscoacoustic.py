import torch
from .utils import diff_using_roll, to_tensor

def _time_step(vx, vz, p, r, vp, rho, Q, omega, b=None, dt=1.0, dh=1.0):

    Kappa = rho*vp**2
    t_sigma = omega**-1*(torch.sqrt(1+(Q**-2))-Q**-1)
    t_epslion = (omega**2 * t_sigma)**-1
    t = t_epslion/(t_sigma-1e-8) - 1
    # 更新速度场
    # x -- 2
    # z -- 1
    yvx = vx - rho**-1 * dt * dh**-1 * diff_using_roll(p, dim=2, append=False) - dt * b * vx
    yvz = vz - rho**-1 * dt * dh**-1 * diff_using_roll(p, dim=1, append=False) - dt * b * vz
    nabla = dh ** -1 * (diff_using_roll(yvx, dim=2)+diff_using_roll(yvz, dim=1))

    yr = r - dt * t_sigma**-1 * (Kappa*t*nabla + r) - dt * b * r

    yp = p - dt * Kappa * (t+1) * nabla - 0.5* dt * (yr+r) - dt * b * p

    return yvx, yvz, yp, yr


class TimeStep(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vx, vz, p, r,
                vp, rho, Q,
                dt, h, d, t, it, omega):

        # lame_lambda = rho*(vp.pow(2)-2*vs.pow(2))
        # lame_mu = rho*(vs.pow(2))
        # ctx.models = [vp, rho, Q]
        # ctx.paras = [omega, dt, h, d]
        # ctx.save_gpu = it > 1

        # if t%it==0:
        #     ctx.save_for_backward(vx, vz)

        yvx, yvz, yp, yr = _time_step(vx, vz, p, r, 
                                      vp, rho, Q, omega, d, dt, h)

        return yvx, yvz, yp, yr

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
        vx, vz, p, r: 
             wave field one time step ago
        vp  :
            Vp vpocity.
        vs  :
            Vs vpocity.
        rho : 
            Projected density, required for nonlinear response (this gets passed in to avoid generating it on each time step, saving memory for backprop)
        """
        t = kwargs['t']
        it = kwargs['it']
        omega = kwargs['omega']
        vx, vz, p, r = wavefields
        vp, rho, Q = model_vars

        hidden = TimeStep.apply(vx, vz, p, r, vp, rho, Q,
                                self.dt, self.geom.h, self.geom.b, t, it, omega)

        return hidden
