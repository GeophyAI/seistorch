import torch
from torch.autograd.functional import hessian

# Define a function that takes vp and vs as inputs and returns a scalar value
def my_function(vp, vs):
    # Define your function here
    # For example, you can return the sum of vp and vs
    return torch.sum(vp + vs)

# Create input parameters
vp = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
vs = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

# Calculate diagonal Hessian matrices
hessian_vp = hessian(lambda x: my_function(x, vs), vp, create_graph=True)
hessian_vs = hessian(lambda x: my_function(vp, x), vs, create_graph=True)

# Extract diagonal elements
hessian_vp_diag = torch.diagonal(hessian_vp)
hessian_vs_diag = torch.diagonal(hessian_vs)

print("Hessian_vp:", hessian_vp)
print("Hessian_vs:", hessian_vs)
print("Hessian_vp_diag:", hessian_vp_diag)
print("Hessian_vs_diag:", hessian_vs_diag)
