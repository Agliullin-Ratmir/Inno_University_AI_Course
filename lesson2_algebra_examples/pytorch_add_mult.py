import torch
A = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

B = torch.tensor([[9, 8, 7],
                  [6, 5, 4],
                  [3, 2, 1]])
sum_result = A + B
print("Summation:\n", sum_result)
product_result = torch.matmul(A, B)
print("Multiplication:\n", product_result)

