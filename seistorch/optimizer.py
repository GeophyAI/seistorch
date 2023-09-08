import numpy as np
import torch, math
import torch.distributed as dist

from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional


class SophiaG(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), rho = 0.04,
         weight_decay=1e-1, *, maximize: bool = False,
         capturable: bool = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= rho:
            raise ValueError("Invalid rho parameter at index 1: {}".format(rho))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, rho=rho, 
                        weight_decay=weight_decay, 
                        maximize=maximize, capturable=capturable)
        super(SophiaG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
            group.setdefault('capturable', False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))
    
    @torch.no_grad()
    def update_hessian(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                if 'hessian' not in state.keys():
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state['hessian'].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)


    @torch.no_grad()
    def step(self, closure=None, bs=5120):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            state_steps = []
            hessian = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                
                if p.grad.is_sparse:
                    raise RuntimeError('Hero does not support sparse gradients')
                grads.append(p.grad)
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                if 'hessian' not in state.keys():
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)                

                exp_avgs.append(state['exp_avg'])
                state_steps.append(state['step'])
                hessian.append(state['hessian'])
                
                if self.defaults['capturable']:
                    bs = torch.ones((1,), dtype=torch.float, device=p.device) * bs

            sophiag(params_with_grad,
                  grads,
                  exp_avgs,
                  hessian,
                  state_steps,
                  bs=bs,
                  beta1=beta1,
                  beta2=beta2,
                  rho=group['rho'],
                  lr=group['lr'],
                  weight_decay=group['weight_decay'],
                  maximize=group['maximize'],
                  capturable=group['capturable'])

        return loss

def sophiag(params: List[Tensor],
          grads: List[Tensor],
          exp_avgs: List[Tensor],
          hessian: List[Tensor],
          state_steps: List[Tensor],
          capturable: bool = False,
          *,
          bs: int,
          beta1: float,
          beta2: float,
          rho: float,
          lr: float,
          weight_decay: float,
          maximize: bool):

    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    
    func = _single_tensor_sophiag

    func(params,
         grads,
         exp_avgs,
         hessian,
         state_steps,
         bs=bs,
         beta1=beta1,
         beta2=beta2,
         rho=rho,
         lr=lr,
         weight_decay=weight_decay,
         maximize=maximize,
         capturable=capturable)

def _single_tensor_sophiag(params: List[Tensor],
                         grads: List[Tensor],
                         exp_avgs: List[Tensor],
                         hessian: List[Tensor],
                         state_steps: List[Tensor],
                         *,
                         bs: int,
                         beta1: float,
                         beta2: float,
                         rho: float,
                         lr: float,
                         weight_decay: float,
                         maximize: bool,
                         capturable: bool):

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        hess = hessian[i]
        step_t = state_steps[i]

        if capturable:
            assert param.is_cuda and step_t.is_cuda and bs.is_cuda 
            
        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            hess = torch.view_as_real(hess)
            param = torch.view_as_real(param)

        # update step
        step_t += 1

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        
        if capturable:
            step = step_t
            step_size = lr 
            step_size_neg = step_size.neg()

            ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(None,1)
            param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)
        else:
            step = step_t.item()
            step_size_neg = - lr 
            
            ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(None,1)
            param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)

def gram_schmidt_orthogonalization(grad_vp, grad_vs):
    """
    对梯度向量grad_vp和grad_vs进行Gram-Schmidt正交化处理。
    """
    # 计算grad_vp和grad_vs之间的内积
    inner_product = np.dot(grad_vp.flatten(), grad_vs.flatten())

    # 计算grad_vp的模平方
    norm_vp_squared = np.dot(grad_vp.flatten(), grad_vp.flatten())

    # 计算正交化后的grad_vs
    orthogonalized_grad_vs = grad_vs - (inner_product / norm_vp_squared) * grad_vp

    return grad_vp, orthogonalized_grad_vs


class NonlinearConjugateGradient(Optimizer):
    def __init__(self, params, lr=1, beta_type='FR', tol=1e-7, c1=1e-4, c2=0.9, max_iter_line_search=50):
        """
        初始化优化器参数

        :param params: 模型参数
        :param lr: 初始步长
        :param beta_type: 更新beta的方法，有'FR' (Fletcher-Reeves) 和 'PR' (Polak-Ribiere) 两种选择
        :param tol: 共轭梯度搜索的终止条件（当梯度范数小于tol时）
        :param c1: 线搜索的Wolfe条件参数1
        :param c2: 线搜索的Wolfe条件参数2
        :param max_iter_line_search: 线搜索的最大迭代次数
        """
        defaults = dict(lr=lr, beta_type=beta_type, tol=tol, c1=c1, c2=c2, max_iter_line_search=max_iter_line_search)
        super(NonlinearConjugateGradient, self).__init__(params, defaults)

    def _line_search(self, closure, current_loss, p, grad, direction, lr, c1, c2, max_iter):
        """
        执行Wolfe线搜索

        :param closure: 一个可调用对象，用于重新计算模型的损失值和梯度
        :param p: 模型参数
        :param grad: 当前梯度
        :param direction: 搜索方向
        :param lr: 初始步长
        :param c1: 线搜索的Wolfe条件参数1
        :param c2: 线搜索的Wolfe条件参数2
        :param max_iter: 线搜索的最大迭代次数
        """

        armijo_condition_met = False
        wolfe_condition_met = False
        iter_count = 0
        lr *= 1/(direction.abs().max().item())
        while (not armijo_condition_met or not wolfe_condition_met) and iter_count < max_iter:
            p.data.add_(lr * direction)  # 更新参数
            with torch.no_grad():
                new_loss = closure().item()
            p.data.sub_(lr * direction)  # 撤销更新
            print(f"Performing line search {iter_count}/{max_iter}, loss:{new_loss}v.s.{current_loss}")

            armijo_condition_met = new_loss <= current_loss + c1 * lr * grad.flatten().dot(direction.flatten())
            wolfe_condition_met = grad.flatten().dot(direction.flatten()) >= c2 * direction.flatten().dot(direction.flatten())

            if not armijo_condition_met or not wolfe_condition_met:
                lr *= 0.5  # 减小步长

            iter_count += 1

        return lr
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        进行一步非线性共轭梯度优化

        :param closure: 一个可调用对象，用于重新计算模型的损失值和梯度
        """
        if closure is not None:
            closure = torch.enable_grad()(closure)
            loss = closure()
        else:
            raise RuntimeError("非线性共轭梯度优化器需要一个闭包函数以重新计算梯度")

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['prev_grad'] = torch.zeros_like(grad)
                    state['direction'] = -grad.clone()

                prev_grad = state['prev_grad']
                direction = state['direction']

                # 计算beta值
                if prev_grad.norm()==0:
                    beta = 0
                elif group['beta_type'] == 'FR':
                    beta = grad.norm()**2 / prev_grad.norm()**2
                elif group['beta_type'] == 'PR':
                    beta = (grad - prev_grad).flatten().dot(grad.flatten()) / prev_grad.norm()**2
                else:
                    raise ValueError("无效的beta_type参数，可选值为'FR'或'PR'")

                # 更新搜索方向
                direction = -grad + beta * direction

                # 使用线搜索更新模型参数
                lr = self._line_search(closure, loss.item(), p, grad, direction, group['lr'], group['c1'], group['c2'], group['max_iter_line_search'])
                p.data.add_(lr * direction)

                # 更新状态
                state['step'] += 1
                state['prev_grad'] = grad.clone()
                state['direction'] = direction

                # 终止条件
                if grad.norm() <= group['tol']:
                    break

        return loss


class Adahessian(Optimizer):
    """Implements Adahessian algorithm.
    It has been proposed in `ADAHESSIAN: An Adaptive Second Order Optimizer for Machine Learning`.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 0.15)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-4)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        hessian_power (float, optional): Hessian power (default: 1). You can also try 0.5. For some tasks we found this to result in better performance.
        single_gpu (Bool, optional): Do you use distributed training or not "torch.nn.parallel.DistributedDataParallel" (default: True)
    """

    def __init__(self, params, lr=0.15, betas=(0.9, 0.999), eps=1e-4,
                 weight_decay=0, hessian_power=1, single_gpu=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(
                    betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(
                    betas[1]))
        if not 0.0 <= hessian_power <= 1.0:
            raise ValueError("Invalid Hessian power value: {}".format(hessian_power))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, hessian_power=hessian_power)
        self.single_gpu = single_gpu 
        super(Adahessian, self).__init__(params, defaults)

    def get_trace(self, params, grads):
        """
        compute the Hessian vector product with a random vector v, at the current gradient point,
        i.e., compute the gradient of <gradsH,v>.
        :param gradsH: a list of torch variables
        :return: a list of torch tensors
        """

        # Check backward was called with create_graph set to True
        for i, grad in enumerate(grads):
            if grad.grad_fn is None:
                raise RuntimeError('Gradient tensor {:} does not have grad_fn. When calling\n'.format(i) +
                           '\t\t\t  loss.backward(), make sure the option create_graph is\n' +
                           '\t\t\t  set to True.')

        v = [2 * torch.randint_like(p, high=2) - 1 for p in params]

        # this is for distributed setting with single node and multi-gpus, 
        # for multi nodes setting, we have not support it yet.
        if not self.single_gpu:
            for v1 in v:
                dist.all_reduce(v1)
        if not self.single_gpu:
            for v_i in v:
                v_i[v_i < 0.] = -1.
                v_i[v_i >= 0.] = 1.

        hvs = torch.autograd.grad(
            grads,
            params,
            grad_outputs=v,
            only_inputs=True,
            retain_graph=True)

        hutchinson_trace = []
        for hv in hvs:
            param_size = hv.size()
            if len(param_size) <= 2:  # for 0/1/2D tensor
                # Hessian diagonal block size is 1 here.
                # We use that torch.abs(hv * vi) = hv.abs()
                tmp_output = hv.abs()

            elif len(param_size) == 4:  # Conv kernel
                # Hessian diagonal block size is 9 here: torch.sum() reduces the dim 2/3.
                # We use that torch.abs(hv * vi) = hv.abs()
                tmp_output = torch.mean(hv.abs(), dim=[2, 3], keepdim=True)
            hutchinson_trace.append(tmp_output)

        # this is for distributed setting with single node and multi-gpus, 
        # for multi nodes setting, we have not support it yet.
        if not self.single_gpu:
            for output1 in hutchinson_trace:
                dist.all_reduce(output1 / torch.cuda.device_count())
        
        return hutchinson_trace

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            gradsH: The gradient used to compute Hessian vector product.
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        params = []
        groups = []
        grads = []

        # Flatten groups into lists, so that
        #  hut_traces can be called with lists of parameters
        #  and grads 
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    params.append(p)
                    groups.append(group)
                    grads.append(p.grad)

        # get the Hessian diagonal

        hut_traces = self.get_trace(params, grads)

        for (p, group, grad, hut_trace) in zip(params, groups, grads, hut_traces):

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data)
                # Exponential moving average of Hessian diagonal square values
                state['exp_hessian_diag_sq'] = torch.zeros_like(p.data)

            exp_avg, exp_hessian_diag_sq = state['exp_avg'], state['exp_hessian_diag_sq']

            beta1, beta2 = group['betas']

            state['step'] += 1

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad.detach_(), alpha=1 - beta1)
            exp_hessian_diag_sq.mul_(beta2).addcmul_(hut_trace, hut_trace, value=1 - beta2)

            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']

            # make the square root, and the Hessian power
            k = group['hessian_power']
            denom = (
                (exp_hessian_diag_sq.sqrt() ** k) /
                math.sqrt(bias_correction2) ** k).add_(
                group['eps'])

            # make update
            p.data = p.data - \
                group['lr'] * (exp_avg / bias_correction1 / denom + group['weight_decay'] * p.data)

        return loss