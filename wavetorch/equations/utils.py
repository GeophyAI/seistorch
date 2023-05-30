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

# def restore_boundaries(tensor, memory, NPML=0, N=70):

#     top, bottom, left, right = memory
#     # tensor = tensor.clone()
#     # tensor[..., NPML:NPML+N, :] = top
#     # tensor[..., -(NPML+N):-NPML, :] = bottom
#     # tensor[..., NPML:NPML+N] = left
#     # tensor[..., -(NPML+N):-NPML] = right

#     return tensor


def save_boundaries(tensor: torch.Tensor, NPML: int=48, N: int=2):
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

def restore_boundaries(tensor, memory, NPML=48, N=2):

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

# def restore_boundaries(tensor, memory, NPML=50, N=20):
#     top, bottom, left, right = memory

#     top = top.unsqueeze(0)
#     bottom = bottom.unsqueeze(0)
#     left = left.unsqueeze(0)
#     right = right.unsqueeze(0)


#     # 创建条件掩码并扩展到与tensor相同的大小
#     condition_top = torch.zeros(top.shape).bool()
#     condition_top = condition_top.expand_as(tensor)
#     condition_bottom = torch.zeros(bottom.shape).bool()
#     condition_bottom = condition_bottom.expand_as(tensor)
#     condition_left = torch.zeros(left.shape).bool()
#     condition_left = condition_left.expand_as(tensor)
#     condition_right = torch.zeros(right.shape).bool()
#     condition_right = condition_right.expand_as(tensor)

#     # 扩展边界数据到与tensor相同的大小
#     top = top.expand_as(tensor)
#     bottom = bottom.expand_as(tensor)
#     left = left.expand_as(tensor)
#     right = right.expand_as(tensor)

#     # 使用 torch.where 更新 tensor
#     tensor = torch.where(condition_top, top, tensor)
#     tensor = torch.where(condition_bottom, bottom, tensor)
#     tensor = torch.where(condition_left, left, tensor)
#     tensor = torch.where(condition_right, right, tensor)

#     return tensor


# def restore_boundaries(tensor, memory, NPML=50, N=2):
#     top, bottom, left, right = memory

#     left = left.unsqueeze(0).requires_grad_()
#     right = right.unsqueeze(0).requires_grad_()
#     left = left[..., NPML+N:-(NPML+N), :]
#     right = right[..., NPML+N:-(NPML+N), :]
#     # 在内部的 tensor 中添加 left 和 right 边界
#     tensor_interior = tensor[..., NPML+N:-(NPML+N), NPML+N:-(NPML+N)]
#     tensor_interior = torch.cat([left, tensor_interior, right], dim=-1)
    
#     top = top.unsqueeze(0).requires_grad_()
#     bottom = bottom.unsqueeze(0).requires_grad_()

#     tensor_top_bottom = torch.cat([top, tensor_interior, bottom], dim=-2)

#     return tensor_top_bottom
