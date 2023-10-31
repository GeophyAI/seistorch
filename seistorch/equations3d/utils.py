import torch

# def save_boundaries(tensor: torch.Tensor, NPML: int=49, N: int=1):
#     """Boundary saving.

#     Args:
#         tensor (torch.Tensor): The wavefield need to be saved (3D).
#         NPML (int): The width of the pml boundary
#         N (int): The diff order (1)

#     Returns:
#         Tuple: top, bottom, left and right boundary.
#     """
#     tensor = tensor.squeeze(0)
#     top = tensor[NPML:NPML+N, ...].clone()
#     bottom = tensor[-N:, ...].clone() if NPML == 0 else tensor[-(NPML+N):-NPML, ...].clone()
#     left = tensor[:,NPML:NPML+N,:].clone()
#     right = tensor[:, -N:, :].clone() if NPML == 0 else tensor[:, -(NPML+N):-NPML,:].clone()
#     front = tensor[..., NPML:NPML+N].clone()
#     back = tensor[..., -N:].clone() if NPML == 0 else tensor[..., -(NPML+N):-NPML].clone()

#     return top, bottom, left, right, front, back

# def restore_boundaries(tensor, memory, NPML=49, N=1):

#     top, bottom, left, right, front, back = memory

#     # Top
#     tensor[0, NPML:NPML+N, ...] = top

#     # Bottom
#     if NPML!=0:
#         tensor[0, -(NPML+N):-NPML, ...] = bottom
#     else:
#         tensor[0, -N:, ...] = bottom

#     # Left
#     tensor[0, :, NPML:NPML+N, :] = left

#     # Right
#     if NPML!=0:
#         tensor[0, :, -(NPML+N):-NPML, :] = right
#     else:
#         tensor[..., :, -N: ] = right

#     # Front
#     tensor[..., NPML:NPML+N] = front

#     # Back
#     if NPML!=0:
#         tensor[..., -(NPML+N):-NPML] = back
#     else:
#         tensor[..., -N:] = back
    
#     return tensor

def save_boundaries(tensor: torch.Tensor, NPML: int=49, N: int=1):
    """Boundary saving.

    Args:
        tensor (torch.Tensor): The wavefield need to be saved (3D).
        NPML (int): The width of the pml boundary
        N (int): The diff order (1)

    Returns:
        Tuple: top, bottom, left and right boundary.
    """
    #tensor = tensor.squeeze(0)
    # Shape: (batch, nx, nz, ny)
    cpu = torch.device("cpu")
    nb = True
    top = tensor[:, NPML:NPML+N, ...].clone().to(cpu, non_blocking=nb)#cpu()
    bottom = tensor[:, -N:, ...].clone() if NPML == 0 else tensor[:,-(NPML+N):-NPML, ...].clone().to(cpu, non_blocking=nb)#cpu()
    left = tensor[:,:,NPML:NPML+N,:].clone().to(cpu, non_blocking=nb)#cpu()
    right = tensor[:, :, -N:, :].clone() if NPML == 0 else tensor[:, :, -(NPML+N):-NPML,:].clone().to(cpu, non_blocking=nb)#cpu()
    front = tensor[..., NPML:NPML+N].clone().to(cpu, non_blocking=nb)#cpu()
    back = tensor[..., -N:].clone() if NPML == 0 else tensor[..., -(NPML+N):-NPML].clone().to(cpu, non_blocking=nb)#cpu()

    return top, bottom, left, right, front, back

def restore_boundaries(tensor, memory, NPML=49, N=1):
    device = tensor.device

    top, bottom, left, right, front, back = memory

    # Top
    # Shape: (batch, nx, nz, ny)
    tensor[:, NPML:NPML+N, ...] = top.to(device)

    # Bottom
    if NPML!=0:
        tensor[:, -(NPML+N):-NPML, ...] = bottom.to(device)
    else:
        tensor[:, -N:, ...] = bottom.to(device)

    # Left
    tensor[:, :, NPML:NPML+N, :] = left.to(device)

    # Right
    if NPML!=0:
        tensor[:, :, -(NPML+N):-NPML, :] = right.to(device)
    else:
        tensor[:, :, -N:,:] = right.to(device)

    # Front
    tensor[..., NPML:NPML+N] = front.to(device)

    # Back
    if NPML!=0:
        tensor[..., -(NPML+N):-NPML] = back.to(device)
    else:
        tensor[..., -N:] = back.to(device)
    
    return tensor