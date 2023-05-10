import torch
from torch.optim import ConjugateGradient

# 定义一个简单的函数
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

def f(x):
    return x[0] ** 2 + x[1] * x[2] + x[2] ** 3

# 计算梯度
y = f(x)
y.backward(create_graph=True)

hessian_diag = []
for i in range(len(x)):
    grad_i = x.grad[i]
    x.grad.data.zero_()  # 必须清空梯度，否则梯度会累积
    if grad_i.requires_grad:
        hessian_i = torch.autograd.grad(grad_i, x, retain_graph=True, create_graph=True)[0]
        hessian_diag.append(hessian_i[i].item())
    else:
        hessian_diag.append(0)

print('Hessian diagonal: ', hessian_diag)
