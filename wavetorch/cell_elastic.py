import torch
from .utils import to_tensor

NEW_DIFF = True

def saturable_damping(u, uth, b0):
    return b0 / (1 + torch.abs(u / uth).pow(2))

def diff_using_roll(input, dim=-1, append=True, padding_value=0):

    dim = input.dim() + dim if dim < 0 else dim
    shifts = -1 if append else 1
    rolled_input = torch.roll(input, shifts=shifts, dims=dim)

    # Fill the idex with value padding_value
    index = [slice(None)] * input.dim()
    index[dim] = -1 if append else 0
    rolled_input[tuple(index)] = padding_value

    diff_result = rolled_input - input if append else input-rolled_input
    return diff_result
    
def _time_step_vel(rho, vx, vz, txx, tzz, txz, dt, h, d):
    
    nz, nx = rho.shape
    if NEW_DIFF:
        txx_x = diff_using_roll(txx, 2)
        txz_z = diff_using_roll(txz, 1, False)
        tzz_z = diff_using_roll(tzz, 1, False)
        txz_x = diff_using_roll(txz, 2, False)
    else:
        txx_x = torch.diff(txx, append=(torch.zeros(1, nz, 1).to(txx.device)), dim=2)
        txz_z = torch.diff(txz, prepend=(torch.zeros(1, 1, nx).to(txx.device)), dim=1)
        tzz_z = torch.diff(tzz, prepend=(torch.zeros(1, 1, nx).to(txx.device)), dim=1)
        txz_x = torch.diff(txz, prepend=(torch.zeros(1, nz, 1).to(txx.device)), dim=2)

    # c = a+d*b**-1
    c = 0.5*dt*d
    # Update y_vx
    y_vx = (1+c)**-1*(dt*rho.pow(-1)*h.pow(-1)*(txx_x+txz_z)+(1-c)*vx)
    # Update y_vz
    y_vz = (1+c)**-1*(dt*rho.pow(-1)*h.pow(-1)*(txz_x+tzz_z)+(1-c)*vz)

    return y_vx, y_vz

def _time_step_stress(lame_lambda, lame_mu, vx, vz, txx, tzz, txz, dt, h, d):
    
    nz, nx = lame_lambda.shape

    if NEW_DIFF:
        vx_x = diff_using_roll(vx, 2, False)
        vz_z = diff_using_roll(vz, 1)
        vx_z = diff_using_roll(vx, 1)
        vz_x = diff_using_roll(vz, 2)
    else:
        vx_x = torch.diff(vx, prepend=(torch.zeros(1, nz, 1).to(txx.device)), dim=2)
        vz_z = torch.diff(vz, append=(torch.zeros(1, 1, nx).to(txx.device)), dim=1)
        vx_z = torch.diff(vx, append=(torch.zeros(1, 1, nx).to(txx.device)), dim=1)
        vz_x = torch.diff(vz, append=(torch.zeros(1, nz, 1).to(txx.device)), dim=2)
    # c = (1+0.5*dt*d)**-1
    c = 0.5*dt*d
    # Equation A-8
    y_txx  = (1+c)**-1*(dt*h.pow(-1)*((lame_lambda+2*lame_mu)*vx_x+lame_lambda*vz_z)+(1-c)*txx)
    # Equation A-9
    y_tzz  = (1+c)**-1*(dt*h.pow(-1)*((lame_lambda+2*lame_mu)*vz_z+lame_lambda*vx_x)+(1-c)*tzz)
    # Equation A-10
    y_txz = (1+c)**-1*(dt*lame_mu*h.pow(-1)*(vz_x+vx_z)+(1-c)*txz)
    
    return y_txx, y_tzz, y_txz


class TimeStep(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vx, vz, txx, tzz, txz,
                vp, vs, rho,
                dt, h, d, t, it):
        
        # lame_lambda = rho*(vp.pow(2)-2*vs.pow(2))
        # lame_mu = rho*(vs.pow(2))
        ctx.models = [vp, vs, rho]
        ctx.paras = [dt, h, d]
        ctx.save_gpu = it > 1

        if t%it==0:
            ctx.save_for_backward(vx, vz)

        y_txx, y_tzz, y_txz = _time_step_stress(rho*(vp.pow(2)-2*vs.pow(2)),  rho*(vs.pow(2)), vx, vz, txx, tzz, txz, dt, h, d)
        y_vx, y_vz = _time_step_vel(rho, vx, vz, y_txx, y_tzz, y_txz, dt, h, d)

        return y_vx, y_vz, y_txx, y_tzz, y_txz
    
    @staticmethod
    def backward(ctx, grad_y_vx, grad_y_vz, grad_y_txx, grad_y_tzz, grad_y_txz):

        vp, vs, rho = ctx.models
        dt, h, d = ctx.paras
        # If save_gpu is True and cannot access to ctx.saved_tensors,
        # Then the gradient of model parameters are all set to zeros.
        # If not save_gpu, or save_gpu and the ctx.saved_tensors can
        # be accessed, then calculate the gradient
        if ctx.save_gpu and (not ctx.saved_tensors):
            _o = torch.zeros_like(grad_y_vx)
        elif (not ctx.save_gpu) or (ctx.save_gpu and ctx.saved_tensors):
            vx, vz = ctx.saved_tensors
            nz, nx = vp.shape
            device = rho.device

            # x---dim=2 
            # z---dim=1
            # prepend--- False
            # append---  True
            if NEW_DIFF:
                vx_x = diff_using_roll(vx, 2, False)
                vx_z = diff_using_roll(vx, 1)

                vz_z = diff_using_roll(vz, 1)
                vz_x = diff_using_roll(vz, 2)
            else:
                vx_x = torch.diff(vx, prepend=(torch.zeros(1, nz, 1).to(device)), dim=2)
                vx_z = torch.diff(vx, append=(torch.zeros(1, 1, nx).to(device)), dim=1)

                vz_z = torch.diff(vz, append=(torch.zeros(1, 1, nx).to(device)), dim=1)
                vz_x = torch.diff(vz, append=(torch.zeros(1, nz, 1).to(device)), dim=2)


        grad_rho = grad_vp = grad_vs = None
        grad_vx = grad_vz = grad_txx = grad_tzz = grad_txz = grad_dt = grad_h = grad_d = None
        c = 0.5*dt*d
        lame_lambda = rho*(vp.pow(2)-2*vs.pow(2))
        lame_mu = rho*(vs.pow(2))
        device = rho.device
        nz, nx = vp.shape
        if NEW_DIFF:
            grad_y_txx_x = diff_using_roll(grad_y_txx, 2)
            grad_y_tzz_z = diff_using_roll(grad_y_tzz, 1, False)

            grad_y_txz_x = diff_using_roll(grad_y_txz, 2, False)
            grad_y_txz_z = diff_using_roll(grad_y_txz, 1, False)
            grad_y_txx_z = diff_using_roll(grad_y_txx, 1, False)
            grad_y_tzz_x = diff_using_roll(grad_y_tzz, 2)
        else:
            grad_y_txx_x = torch.diff(grad_y_txx, append=(torch.zeros(1, nz, 1).to(device)), dim=2)
            grad_y_tzz_z = torch.diff(grad_y_tzz, prepend=(torch.zeros(1, 1, nx).to(device)), dim=1)
            grad_y_txz_x = torch.diff(grad_y_txz, prepend=(torch.zeros(1, nz, 1).to(device)), dim=2)
            grad_y_txz_z = torch.diff(grad_y_txz, prepend=(torch.zeros(1, 1, nx).to(device)), dim=1)
            
            grad_y_txx_z = torch.diff(grad_y_txx, prepend=(torch.zeros(1, 1, nx).to(device)), dim=1)
            grad_y_tzz_x = torch.diff(grad_y_tzz, append=(torch.zeros(1, nz, 1).to(device)), dim=2)

        if ctx.needs_input_grad[0]: # vx
            grad_vx = (1+c)**-1*(dt*h**-1*rho**-1*((lame_lambda+2*lame_mu)*grad_y_txx_x\
                                                   +lame_mu*grad_y_txz_z+lame_lambda*grad_y_tzz_x)\
                    +(1-c)*grad_y_vx)
            grad_vx_x = torch.diff(grad_vx, prepend=(torch.zeros(1, nz, 1).to(device)), dim=2)
            grad_vx_z = torch.diff(grad_vx, append=(torch.zeros(1, 1, nx).to(device)), dim=1)

        if ctx.needs_input_grad[1]: # vz
            grad_vz = (1+c)**-1*(dt*h**-1*rho**-1*((lame_lambda+2*lame_mu)*grad_y_tzz_z\
                                                   +lame_mu*grad_y_txz_x+lame_lambda*grad_y_txx_z)\
                    +(1-c)*grad_y_vz)
            grad_vz_x = torch.diff(grad_vz, append=(torch.zeros(1, nz, 1).to(device)), dim=2)
            grad_vz_z = torch.diff(grad_vz, append=(torch.zeros(1, 1, nx).to(device)), dim=1)
            
        if ctx.needs_input_grad[2]: # txx
            grad_txx = (1+c)**-1*(dt*h**-1*grad_vx_x+(1-c)*grad_y_txx)

        if ctx.needs_input_grad[3]: # tzz
            grad_tzz = (1+c)**-1*(dt*h**-1*grad_vz_z+(1-c)*grad_y_tzz)

        if ctx.needs_input_grad[4]: # txz
            grad_txz = (1+c)**-1*(dt*h**-1*(grad_vz_x+grad_vx_z)+(1-c)*grad_y_txz)

        if ctx.save_gpu and (not ctx.saved_tensors):
            return grad_vx, grad_vz, grad_txx, grad_tzz, grad_txz, _o, _o, _o, None, None, None, None, None

        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1] and ctx.needs_input_grad[5]: # Vp
            grad_lambda = -(vx_x+vz_z)*(grad_txx+grad_tzz)*h**-1
            grad_mu = -(2*(grad_y_txx*vx_x+grad_tzz*vz_z)+grad_y_txz*(vz_x+vx_z))
            grad_vp = 2*rho*vp*grad_lambda
            
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1] and ctx.needs_input_grad[6]: # Vs
            grad_vs = -4*rho*vs*grad_lambda+2*rho*vs*grad_mu

        if ctx.needs_input_grad[7]: # Rho
            pass

        return grad_vx, grad_vz, grad_txx, grad_tzz, grad_txz, grad_vp, grad_vs, grad_rho, grad_dt, grad_h, grad_d, None, None

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

    def forward(self, vx, vz, txx, tzz, txz, model_vars: tuple, t, it):
        """Take a step through time
        Parameters
        ----------
        vx, vz, txx, tzz, txz : 
             wave field one time step ago (hidden state)
        vp  :
            Vp velocity.
        vs  :
            Vs velocity.
        rho : 
            Projected density, required for nonlinear response (this gets passed in to avoid generating it on each time step, saving memory for backprop)
        """
        vp, vs, rho = model_vars
        hidden = TimeStep.apply(vx, vz, txx, tzz, txz, vp, vs, rho, 
                                self.dt, self.geom.h, self.geom.d, t, it)

        return hidden