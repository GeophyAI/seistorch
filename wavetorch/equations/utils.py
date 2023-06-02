import torch

def diff_using_roll(input, dim=-1, forward=True, padding_value=0):

    def forward_diff(x, dim=-1, padding_value=0):
        """
        Compute the forward difference of an input tensor along a given dimension.

        Args:
            x (torch.Tensor): Input tensor.
            dim (int, optional): The dimension along which to compute the difference.
            padding_value (float, optional): The value to use for padding.

        Returns:
            torch.Tensor: The forward difference of the input tensor.
        """
        diff = x - torch.roll(x, shifts=1, dims=dim)
        diff[..., 0] = padding_value  # pad with specified value
        return diff

    def backward_diff(x, dim=-1, padding_value=0):
        """
        Compute the backward difference of an input tensor along a given dimension.

        Args:
            x (torch.Tensor): Input tensor.
            dim (int, optional): The dimension along which to compute the difference.
            padding_value (float, optional): The value to use for padding.

        Returns:
            torch.Tensor: The backward difference of the input tensor.
        """
        diff = torch.roll(x, shifts=-1, dims=dim) - x
        diff[..., -1] = padding_value  # pad with specified value
        return diff

    if forward:
        return forward_diff(input, dim=dim)
    else:
        return backward_diff(input, dim=dim)

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
    top = tensor[NPML:NPML+N, :].clone()
    #bottom = tensor[-(NPML+N):-NPML, :].clone()
    bottom = tensor[-N: , :].clone() if NPML == 0 else tensor[-(NPML+N):-NPML, :].clone()
    left = tensor[:,NPML:NPML+N].clone()
    #right = tensor[:, -(NPML+N):-NPML].clone()
    right = tensor[:, -N: ].clone() if NPML == 0 else tensor[:, -(NPML+N):-NPML].clone()

    return top, bottom, left, right

def restore_boundaries(tensor, memory, NPML=49, N=1):

    top, bottom, left, right = memory
    #ntensor = tensor.clone()
    tensor[..., NPML:NPML+N, :] = top#.requires_grad_()
    if NPML!=0:
        tensor[..., -(NPML+N):-NPML, :] = bottom#.requires_grad_()
    else:
        tensor[..., -N: , :] = bottom
    tensor[..., NPML:NPML+N] = left#.requires_grad_()
    if NPML!=0:
        tensor[..., -(NPML+N):-NPML] = right#.requires_grad_()
    else:
        tensor[..., :, -N: ] = right
    
    return tensor