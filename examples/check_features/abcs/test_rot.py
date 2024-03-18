import torch

# 创建示例非方形矩阵
array = torch.randn(5, 3)  # 假设 nx=3, nz=5

# 在第一个维度上进行旋转
rotated_array_dim0 = torch.rot90(array, k=1, dims=(0, 1))

# 在第二个维度上进行旋转
rotated_array_dim1 = torch.rot90(array, k=1, dims=(1, 0))

# 打印结果
print("原始数组:")
print(array)
print("在第一个维度上旋转90度后的数组:")
print(rotated_array_dim0)
print("在第二个维度上旋转90度后的数组:")
print(rotated_array_dim1)
