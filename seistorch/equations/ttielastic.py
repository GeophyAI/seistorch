from .utils import diff_using_roll, restore_boundaries
import torch

NPML = 49
N = 1

def _time_step_backward(*args):

    vp, vs, rho, eta, gamma, delta = args[0:6]
    vx, vz, txx, tzz, txz = args[6:11]
    dt, h, d = args[11:14]
    vx_bd, vz_bd, txx_bd, tzz_bd, txz_bd = args[-2]
    src_type, src_func, src_values = args[-1]

    vp = vp.unsqueeze(0)
    vs = vs.unsqueeze(0)
    rho = rho.unsqueeze(0)
    d = d.unsqueeze(0)

    """Update velocity components"""
    lame_lambda = rho*(vp.pow(2)-2*vs.pow(2))
    lame_mu = rho*(vs.pow(2))

    # Define the region where the computation is performed
    compute_region_slice = (slice(None), slice(NPML, -NPML), slice(NPML, -NPML))
    update_region_slice = (slice(None), slice(NPML+N, -NPML-N), slice(NPML+N, -NPML-N))

    # Create a copy of the original tensors
    vx_copy, vz_copy = vx.clone(), vz.clone()

    # Replace the original tensors with their sub-tensors within the computation region
    rho = rho[compute_region_slice]
    vx, vz = vx_copy[compute_region_slice], vz_copy[compute_region_slice]
    d = d[compute_region_slice]

    # The rest of your computation code...
    txx_x = diff_using_roll(txx, 2)[compute_region_slice]
    txz_z = diff_using_roll(txz, 1, False)[compute_region_slice]
    tzz_z = diff_using_roll(tzz, 1, False)[compute_region_slice]
    txz_x = diff_using_roll(txz, 2, False)[compute_region_slice]

    c = 0.5*dt*d

    y_vx = (1+c)**-1*(-dt*rho.pow(-1)*h.pow(-1)*(txx_x+txz_z)+(1-c)*vx)
    y_vz = (1+c)**-1*(-dt*rho.pow(-1)*h.pow(-1)*(txz_x+tzz_z)+(1-c)*vz)

    # Write back the results to the original tensors, but only within the update region
    vx_copy[update_region_slice] = y_vx[(slice(None), slice(N,-N), slice(N,-N))]
    vz_copy[update_region_slice] = y_vz[(slice(None), slice(N,-N), slice(N,-N))]

    # Restore the boundary
    vx_copy = restore_boundaries(vx_copy, vx_bd, NPML, N)
    vz_copy = restore_boundaries(vz_copy, vz_bd, NPML, N)

    # Create a copy of the original tensors
    txx_copy, tzz_copy, txz_copy = txx.clone(), tzz.clone(), txz.clone()

    # Replace the original tensors with their sub-tensors within the computation region
    lame_lambda, lame_mu = lame_lambda[compute_region_slice], lame_mu[compute_region_slice]
    vx, vz = vx_copy[compute_region_slice], vz_copy[compute_region_slice]
    txx, tzz, txz = txx_copy[compute_region_slice], tzz_copy[compute_region_slice], txz_copy[compute_region_slice]

    # The rest of your computation code...
    vx_x = diff_using_roll(vx, 2, False)
    vz_z = diff_using_roll(vz, 1)
    vx_z = diff_using_roll(vx, 1)
    vz_x = diff_using_roll(vz, 2)

    c = 0.5*dt*d
    y_txx  = (1+c)**-1*(-dt*h.pow(-1)*((lame_lambda+2*lame_mu)*vx_x+lame_lambda*vz_z)+(1-c)*txx)
    y_tzz  = (1+c)**-1*(-dt*h.pow(-1)*((lame_lambda+2*lame_mu)*vz_z+lame_lambda*vx_x)+(1-c)*tzz)
    y_txz = (1+c)**-1*(-dt*lame_mu*h.pow(-1)*(vz_x+vx_z)+(1-c)*txz)

    # Write back the results to the original tensors, but only within the update region
    txx_copy[update_region_slice] = y_txx[(slice(None), slice(N,-N), slice(N,-N))]
    tzz_copy[update_region_slice] = y_tzz[(slice(None), slice(N,-N), slice(N,-N))]
    txz_copy[update_region_slice] = y_txz[(slice(None), slice(N,-N), slice(N,-N))]

    # Restore the boundary
    txx_copy = restore_boundaries(txx_copy, txx_bd, NPML, N)
    tzz_copy = restore_boundaries(tzz_copy, tzz_bd, NPML, N)
    txz_copy = restore_boundaries(txz_copy, txz_bd, NPML, N)

    for s_type in src_type:
        source_var = eval(s_type+"_copy")
        # source_var = source_var.clone()
        source_var = src_func(source_var, src_values, -1)

    return vx_copy, vz_copy, txx_copy, tzz_copy, txz_copy

def _time_step(*args):

    #vp, vs, rho, epsilon, gamma, delta = args[0:6]
    rho, c11, c13, c33, c15, c35, c55 = args[0:7]
    vx, vz, txx, tzz, txz = args[7:12]
    dt, h, d = args[12:15]
    
    #delta = 0; epsilon = 0 # elastic
    #d = 0.
    # _theta = torch.FloatTensor([45.]).to(vp.device)
    # theta = torch.deg2rad(_theta)#torch.Tensor([_theta*3.141592653589793/180.]).to(vp.device)
    # # Calculate the thomsen parameters
    # f = 1-vs**2/vp**2
    # cv11 = rho*vp.pow(2)*(1+2*epsilon)
    # # cv13 = rho*vp.pow(2)*torch.sqrt(f*(f+2*delta))-rho*vs.pow(2)
    # cv13 = rho*torch.sqrt(((1+2*delta)*vp**2-vs**2)*(vp**2-vs**2))-rho*vs.pow(2)
    # cv33 = rho*vp.pow(2)
    # cv44 = rho*vs.pow(2)

    # # Compute intermediate values

    # cos_theta2 = torch.pow(torch.cos(theta),2)
    # sin_theta2 = torch.pow(torch.sin(theta),2)
    # sin_2theta = torch.sin(2*theta)
    # sin_2theta_sq = torch.pow(sin_2theta,2)

    # # Compute outputs using intermediate values and PyTorch tensor operations

    # c11 = (cos_theta2*cv11+sin_theta2*cv13)*cos_theta2 \
    #     + (cos_theta2*cv13+sin_theta2*cv33)*sin_theta2 \
    #     + sin_2theta_sq*cv44
    # c13 = (cos_theta2*cv11+sin_theta2*cv13)*sin_theta2 \
    #     + (cos_theta2*cv13+sin_theta2*cv33)*cos_theta2 \
    #     -  sin_2theta_sq*cv44
    # c33 = (sin_theta2*cv11+cos_theta2*cv13)*sin_theta2 \
    #     + (sin_theta2*cv13+cos_theta2*cv33)*cos_theta2 \
    #     +  sin_2theta_sq*cv44
    # c15 = .5*(cos_theta2*cv11+sin_theta2*cv13)*sin_2theta \
    #     - .5*(cos_theta2*cv13+sin_theta2*cv33)*sin_2theta \
    #     - sin_2theta*cv44*(cos_theta2-sin_theta2)
    # c35 = .5*(sin_theta2*cv11+cos_theta2*cv13)*sin_2theta \
    #     - .5*(sin_theta2*cv13+cos_theta2*cv33)*sin_2theta \
    #     + sin_2theta*cv44*(cos_theta2-sin_theta2)
    # c55 = .25*(sin_2theta*cv11-sin_2theta*cv13)*sin_2theta \
    #     - .25*(sin_2theta*cv13-sin_2theta*cv33)*sin_2theta \
    #     + cv44*(cos_theta2-sin_theta2)

    vx_x = diff_using_roll(vx, 2, False)
    vz_z = diff_using_roll(vz, 1)
    vx_z = diff_using_roll(vx, 1)
    vz_x = diff_using_roll(vz, 2)

    c = 0.5*dt*d

    # lame_lambda = rho*(vp.pow(2)-2*vs.pow(2))
    # lame_mu = rho*(vs.pow(2))
    # Equation A-8
    y_txx  = (1+c)**-1*(dt*h.pow(-1)*(c11*vx_x+c13*vz_z+c15*(vz_x+vx_z))+(1-c)*txx)
    # y_txx  = (1+c)**-1*(dt*h.pow(-1)*((lame_lambda+2*lame_mu)*vx_x+lame_lambda*vz_z)+(1-c)*txx)
    # Equation A-9
    # y_tzz  = (1+c)**-1*(dt*h.pow(-1)*(c13*vx_x+c33**vz_z+c35*vz_x+c35*vx_z)+(1-c)*tzz)
    # assert torch.allclose(c33, lame_lambda+2*lame_mu)
    y_tzz  = (1+c)**-1*(dt*h.pow(-1)*(c33*vz_z+c13*vx_x+c35*(vz_x+vx_z))+(1-c)*tzz)


    # Equation A-10
    y_txz = (1+c)**-1*(dt*h.pow(-1)*(c15*vx_x+c35*vz_z+c55*vz_x+c55*vx_z)+(1-c)*txz)
    # y_txz = (1+c)**-1*(dt*h.pow(-1)*(lame_mu*vz_x+lame_mu*vx_z)+(1-c)*txz)


    # assert torch.allclose(_y_txz, y_txz)

    txx_x = diff_using_roll(y_txx, 2)
    txz_z = diff_using_roll(y_txz, 1, False)
    tzz_z = diff_using_roll(y_tzz, 1, False)
    txz_x = diff_using_roll(y_txz, 2, False)

    # Update y_vx
    y_vx = (1+c)**-1*(dt*rho.pow(-1)*h.pow(-1)*(txx_x+txz_z)+(1-c)*vx)
    # Update y_vz
    y_vz = (1+c)**-1*(dt*rho.pow(-1)*h.pow(-1)*(txz_x+tzz_z)+(1-c)*vz)

    return y_vx, y_vz, y_txx, y_tzz, y_txz