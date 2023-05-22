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
    
def restore_boundaries(tensor, memory, NPML=50, N=2):

    top, bottom, left, right = memory
    tensor[..., NPML:NPML+N, :] = top
    tensor[..., -(NPML+N):-NPML, :] = bottom
    tensor[..., NPML:NPML+N] = left
    tensor[..., -(NPML+N):-NPML] = right
    
    return tensor