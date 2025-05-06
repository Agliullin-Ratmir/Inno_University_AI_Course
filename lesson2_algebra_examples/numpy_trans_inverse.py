import numpy as np
A = np.array([[2, 1, 3],
              [1, 0, 2],
              [4, 1, 8]])
A_T = A.T
print("Transposed Matrix:\n", A_T)
A_inv = np.linalg.inv(A)
print("Inversed Matrix:\n", A_inv)

