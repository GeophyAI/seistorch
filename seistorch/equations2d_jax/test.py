import jax.numpy as jnp

class ArrayWrapper:
    def __init__(self, array):
        self.array = array
    
    def __getitem__(self, idx):
        return self.array[idx]
    
    def __setitem__(self, idx, value):
        # 在这里实现类似 `rec.at[:, it, :].set()` 的效果
        self.array = self.array.at[idx].set(value)

# 假设 rec 是一个 JAX 数组
rec = jnp.zeros((10, 10, 10))

# 创建包装类的实例
rec_wrapper = ArrayWrapper(rec)

# 使用自定义类进行赋值，等效于 rec = rec.at[:, it, :].set(u_now[:, 0, recz, pmln:-pmln])
it = 1
u_now = jnp.ones((10, 1, 10, 10))
recz = 5
pmln = 2

# 使用自定义的方式进行赋值
rec_wrapper[:, it, :] = u_now[:, 0, recz, :]

# 你可以通过调用 rec_wrapper.array 来查看更新后的数组
print(rec_wrapper.array)
