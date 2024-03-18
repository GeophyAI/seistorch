import torch

a = torch.rand(2, 1, 4, 4)
src_list = [[0, 0], [1, 1]]

# Unpack src_list into separate arrays for rows and columns
rows, cols = zip(*src_list)

# Create indices tensor
indices = torch.tensor([0, 1])

# Use advanced indexing and broadcasting to increment values
a[indices, :, rows, cols] += 1

print(a)
