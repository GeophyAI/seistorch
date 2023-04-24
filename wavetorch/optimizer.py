import numpy as np

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

# 假设我们已经计算了vp和vs的梯度（grad_vp和grad_vs）
# grad_vp = torch.randn(100, 100)  # 示例：随机生成一个100x100的梯度矩阵
# grad_vs = torch.randn(100, 100)  # 示例：随机生成一个100x100的梯度矩阵

# 使用Gram-Schmidt正交化方法正交化梯度
# orthogonalized_grad_vp, orthogonalized_grad_vs = gram_schmidt_orthogonalization(grad_vp, grad_vs)

# # 根据正交化后的梯度更新模型参数
# vp_updated = vp - alpha_vp * orthogonalized_grad_vp  # alpha_vp为P波速度的学习率
# vs_updated = vs - alpha_vs * orthogonalized_grad_vs  # alpha_vs为S波速度的学习率