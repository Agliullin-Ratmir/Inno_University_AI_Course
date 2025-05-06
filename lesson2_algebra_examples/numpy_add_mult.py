import numpy as np
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

B = np.array([[9, 8, 7],
              [6, 5, 4],
              [3, 2, 1]])
sum_result = A + B
print("Summation:\n", sum_result)
product_result = np.dot(A, B)
print("Multiplication:\n", product_result)

