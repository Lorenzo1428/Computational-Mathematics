import numpy as np

n = 10
A = np.random.rand(n, n)
sum = np.sum(A, axis=0)
print(A, "\n", sum)
