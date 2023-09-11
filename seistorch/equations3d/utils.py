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
    tensor = tensor.squeeze(0)
    top = tensor[NPML:NPML+N, ...].clone().cpu()
    bottom = tensor[-N:, ...].clone() if NPML == 0 else tensor[-(NPML+N):-NPML, ...].clone().cpu()
    left = tensor[:,NPML:NPML+N,:].clone().cpu()
    right = tensor[:, -N:, :].clone() if NPML == 0 else tensor[:, -(NPML+N):-NPML,:].clone().cpu()
    front = tensor[..., NPML:NPML+N].clone().cpu()
    back = tensor[..., -N:].clone() if NPML == 0 else tensor[..., -(NPML+N):-NPML].clone().cpu()

    return top, bottom, left, right, front, back

def restore_boundaries(tensor, memory, NPML=49, N=1):
    device = tensor.device
    #tensor = tensor.cpu()

    top, bottom, left, right, front, back = memory

    # Top
    tensor[0, NPML:NPML+N, ...] = top.to(device)

    # Bottom
    if NPML!=0:
        tensor[0, -(NPML+N):-NPML, ...] = bottom.to(device)
    else:
        tensor[0, -N:, ...] = bottom.to(device)

    # Left
    tensor[0, :, NPML:NPML+N, :] = left.to(device)

    # Right
    if NPML!=0:
        tensor[0, :, -(NPML+N):-NPML, :] = right.to(device)
    else:
        tensor[..., :, -N: ] = right.to(device)

    # Front
    tensor[..., NPML:NPML+N] = front.to(device)

    # Back
    if NPML!=0:
        tensor[..., -(NPML+N):-NPML] = back.to(device)
    else:
        tensor[..., -N:] = back.to(device)
    
    return tensor#.to(device)