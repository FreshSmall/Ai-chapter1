import numpy as np

p = np.random.randn(3, 3)
print(p)
count_greater_than_zero = np.sum(p > 0)
print(f"矩阵中大于0的元素个数:{count_greater_than_zero}")

if __name__ == '__main__':
    pass