import torch
from torch import Tensor
from torch.optim.optimizer import (Optimizer, required, 
                        _use_grad_for_differentiable, 
                        _default_to_fused_or_foreach)
from typing import List, Optional

class Adam(torch.optim.Adam):
   
    @property
    def name(self,):
        return "adam"

    def __init__(self, params, eps=1e-8, **kwargs):
        super(Adam, self).__init__(params, eps=eps)

class Steepestdescent(Optimizer):

    @property
    def name(self,):
        return "sd"

    def __init__(self, params, lr=10.0, maximize: bool = False, differentiable: bool = False, grad_clamp=True, **kwargs):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        
        defaults = dict(lr=lr, 
                        maximize=maximize, 
                        grad_clamp=grad_clamp, 
                        differentiable=differentiable)

        super().__init__(params, defaults)


    # def __setstate__(self, state):
    #     super().__setstate__(state)
    #     for group in self.param_groups:
    #         group.setdefault('maximize', False)
    #         group.setdefault('differentiable', False)
    #         group.setdefault('grad_clamp', False)

    def _init_group(self, group, params_with_grad, d_p_list):
        has_sparse_grad = False

        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                d_p_list.append(p.grad)
                if p.grad.is_sparse:
                    has_sparse_grad = True

                state = self.state[p]

        return has_sparse_grad

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []

            has_sparse_grad = self._init_group(group, params_with_grad, d_p_list)

            self.gd(params_with_grad,
                    d_p_list,
                    lr=group['lr'],
                    grad_clamp=group['grad_clamp'],
                    maximize=group['maximize'],
                    has_sparse_grad=has_sparse_grad)

            # update momentum_buffers in state
            for p in params_with_grad:
                state = self.state[p]

        return loss
        
    def gd(self, params: List[Tensor],
        d_p_list: List[Tensor],
        # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
        # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
        has_sparse_grad: bool = None,
        foreach: Optional[bool] = None,
        *,
        lr: float,
        grad_clamp: bool,
        maximize: bool):
        r"""Functional API that performs SGD algorithm computation.

        See :class:`~torch.optim.SGD` for details.
        """

        if foreach is None:
            # why must we be explicit about an if statement for torch.jit.is_scripting here?
            # because JIT can't handle Optionals nor fancy conditionals when scripting
            if not torch.jit.is_scripting():
                _, foreach = _default_to_fused_or_foreach(params, differentiable=False, use_fused=False)
            else:
                foreach = False

        if foreach and torch.jit.is_scripting():
            raise RuntimeError('torch.jit.script not supported with foreach optimizers')

        if True:
            func = self._single_tensor_gd

        func(params,
            d_p_list,
            lr=lr,
            grad_clamp=grad_clamp,
            has_sparse_grad=has_sparse_grad,
            maximize=maximize)
    
    def _single_tensor_gd(self, params: List[Tensor],
                        d_p_list: List[Tensor],
                        *,
                        lr: float,
                        grad_clamp: bool,
                        maximize: bool,
                        has_sparse_grad: bool):

        for i, param in enumerate(params):
            d_p = d_p_list[i] if not maximize else -d_p_list[i]

            # We need to make sure that \delta m <= lr
            if grad_clamp:
                bound = torch.quantile(d_p, torch.Tensor([0.02, 0.98]).to(d_p.device).to(d_p.dtype))
                d_p = torch.clamp(d_p, min=bound[0], max=bound[1])

            max_value_of_grad = torch.max(torch.abs(d_p.data)).cpu()
            alpha = lr / max_value_of_grad
            param.add_(d_p, alpha=-alpha)

class Cg(Optimizer):

    @property
    def name(self,):
        return "ncg"
    
    def __init__(self, params, lr=1.0, beta_type='PR', grad_clamp=True, **kwargs):
        defaults = dict(lr=lr, 
                        beta_type=beta_type, 
                        grad_clamp=grad_clamp, 
                        clip_value=kwargs['clip_value'])
        super(Cg, self).__init__(params, defaults)

    # def __setstate__(self, state):
    #     super().__setstate__(state)
    #     for group in self.param_groups:
    #         group.setdefault('beta_type', 'FR')
    #         group.setdefault('grad_clamp', True)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # grad = p.grad.data
                # Clamp the gradient
                if group['grad_clamp']:
                    grad_temp = p.grad.data
                    upper = 1-group['clip_value']
                    lower = group['clip_value']
                    print(f'Clip the gradient between {lower} and {upper}')
                    bound = torch.quantile(grad_temp, torch.Tensor([lower, upper]).to(grad_temp.device).to(grad_temp.dtype))
                    grad = torch.clamp(grad_temp, min=bound[0], max=bound[1])
                else:
                    grad = p.grad.data

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['prev_grad'] = torch.zeros_like(grad)
                    state['direction'] = -grad.clone()

                prev_grad = state['prev_grad']
                direction = state['direction']

                # Compute beta
                if prev_grad.norm()==0:
                    beta = 0
                elif group['beta_type'] == 'FR':
                    beta = grad.norm()**2 / prev_grad.norm()**2
                elif group['beta_type'] == 'PR':
                    beta = (grad - prev_grad).flatten().dot(grad.flatten()) / prev_grad.norm()**2
                else:
                    raise ValueError("Invalid beta_type. Must be 'FR' or 'PR'")

                # update direction
                direction = -grad + beta * direction

                # Clamp the direction
                # if group['grad_clamp']:
                #     bound = torch.quantile(direction, torch.Tensor([0.02, 0.98]).to(direction.device).to(direction.dtype))
                #     direction = torch.clamp(direction, min=bound[0], max=bound[1])

                max_value_of_grad = torch.max(torch.abs(direction.data)).cpu()
                alpha = group['lr'] / max_value_of_grad
                p.data.add_(alpha * direction)

                # Update state
                state['step'] += 1
                state['prev_grad'] = grad.clone()
                state['direction'] = direction

        return loss
