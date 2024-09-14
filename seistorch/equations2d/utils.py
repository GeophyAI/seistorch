import torch

# def diff_using_roll(input, dim=-1, forward=True, padding_value=0):

#     def forward_diff(x, dim=-1, padding_value=0):
#         """
#         Compute the forward difference of an input tensor along a given dimension.

#         Args:
#             x (torch.Tensor): Input tensor.
#             dim (int, optional): The dimension along which to compute the difference.
#             padding_value (float, optional): The value to use for padding.

#         Returns:
#             torch.Tensor: The forward difference of the input tensor.
#         """
#         # x[:,0] = padding_value
#         diff = x - torch.roll(x, shifts=1, dims=dim)
#         if dim == 1:
#             diff[:, 0] = padding_value
#         elif dim == 2:
#             diff[..., 0] = padding_value  # pad with specified value
#         return diff

#     def backward_diff(x, dim=-1, padding_value=0):
#         """
#         Compute the backward difference of an input tensor along a given dimension.

#         Args:
#             x (torch.Tensor): Input tensor.
#             dim (int, optional): The dimension along which to compute the difference.
#             padding_value (float, optional): The value to use for padding.

#         Returns:
#             torch.Tensor: The backward difference of the input tensor.
#         """
#         # x[...,-1] = padding_value
#         diff = torch.roll(x, shifts=-1, dims=dim) - x
#         if dim == 1:
#             diff[:, -1] = padding_value
#         elif dim == 2:
#             diff[..., -1] = padding_value  # pad with specified value
#         return diff

#     if forward:
#         return forward_diff(input, dim=dim)
#     else:
#         return backward_diff(input, dim=dim)

def diff_using_roll(input, dim=-1, forward=True, padding_value=0):

    def forward_diff(x, dim, padding_value):
        """
        Compute the forward difference of an input tensor along a given dimension.
        """
        rolled_x = torch.roll(x, shifts=1, dims=dim)
        diff = x - rolled_x
        
        # Create a mask for padding
        pad_mask = torch.zeros_like(x)
        pad_mask_index = [slice(None)] * x.dim()  # Create a list of slices for all dimensions
        pad_mask_index[dim] = 0  # Set the specified dimension to 0 for padding
        pad_mask[tuple(pad_mask_index)] = padding_value
        
        return torch.where(diff != diff, pad_mask, diff)  # Use diff != diff to identify the padded positions

    def backward_diff(x, dim, padding_value):
        """
        Compute the backward difference of an input tensor along a given dimension.
        """
        rolled_x = torch.roll(x, shifts=-1, dims=dim)
        diff = rolled_x - x
        
        # Create a mask for padding
        pad_mask = torch.zeros_like(x)
        pad_mask_index = [slice(None)] * x.dim()  # Create a list of slices for all dimensions
        pad_mask_index[dim] = -1  # Set the specified dimension to -1 for padding
        pad_mask[tuple(pad_mask_index)] = padding_value
        
        return torch.where(diff != diff, pad_mask, diff)  # Use diff != diff to identify the padded positions

    if forward:
        return forward_diff(input, dim, padding_value)
    else:
        return backward_diff(input, dim, padding_value)

def save_boundaries(tensor: torch.Tensor, NPML: int=49, N: int=1):
    """Boundary saving.

    Args:
        tensor (torch.Tensor): The wavefield need to be saved (3D).
        NPML (int): The width of the pml boundary
        N (int): The diff order (1)

    Returns:
        Tuple: top, bottom, left and right boundary.
    """
    tensor = tensor#.squeeze(0)
    top = tensor[:, NPML:NPML+N, :].clone()
    #bottom = tensor[-(NPML+N):-NPML, :].clone()
    bottom = tensor[:, -N: , :].clone() if NPML == 0 else tensor[:, -(NPML+N):-NPML, :].clone()
    left = tensor[...,NPML:NPML+N].clone()
    #right = tensor[:, -(NPML+N):-NPML].clone()
    right = tensor[..., -N: ].clone() if NPML == 0 else tensor[..., -(NPML+N):-NPML].clone()

    return top, bottom, left, right

def restore_boundaries(tensor, memory, NPML=49, N=1, multiple=False):

    top, bottom, left, right = memory

    # For multiple, no need to do this
    if not multiple:
        tensor[:, NPML:NPML+N, :] = top#.requires_grad_()

    if NPML!=0:
        tensor[:, -(NPML+N):-NPML, :] = bottom#.requires_grad_()
    else:
        tensor[:, -N: , :] = bottom

    tensor[..., NPML:NPML+N] = left#.requires_grad_()
    if NPML!=0:
        tensor[..., -(NPML+N):-NPML] = right#.requires_grad_()
    else:
        tensor[..., :, -N: ] = right
    
    return tensor