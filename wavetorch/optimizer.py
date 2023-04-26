import numpy as np
import torch
from torch.optim import Optimizer

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