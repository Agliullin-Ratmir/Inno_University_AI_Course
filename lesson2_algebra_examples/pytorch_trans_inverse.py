import torch
A = torch.tensor([[2.0, 1.0, 3.0],
                  [1.0, 0.0, 2.0],
                  [4.0, 1.0, 8.0]])
A_T = A.T  # OR: A.transpose(0, 1)
print("Transposed Matrix:\n", A_T)
A_inv = torch.linalg.inv(A)
print("Inversed Matrix:\n", A_inv)

