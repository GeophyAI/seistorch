import numpy as np

def find_duplicate_coordinates(ntraces, recs):
    unique_coordinates, unique_indices, counts = np.unique(ntraces, axis=0, return_inverse=True, return_counts=True)

    indices_grouped = np.split(np.argsort(unique_indices), np.cumsum(counts))[:-1]
    recs_grouped = [recs[indices].tolist() for indices in indices_grouped]

    result_dict = dict(zip(map(tuple, unique_coordinates), [{'idx_in_segy': indices.tolist(), 'recs': recs_values} for indices, recs_values in zip(indices_grouped, recs_grouped)]))

    return result_dict

# 示例
ntraces = np.array([
    [1, 2, 3],
    [1, 2, 3],  # 与第一行坐标相同

    [4, 5, 6],
    [7, 8, 9],
    [1, 2, 3],  # 与第一行坐标相同
    [4, 5, 6],  # 与第二行坐标相同
    [10, 11, 12]
])

recs = np.array([101, 10, 102, 103, 104, 105, 106])

result = find_duplicate_coordinates(ntraces, recs)
print(result)
