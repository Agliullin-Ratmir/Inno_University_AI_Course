import torch

# Create two example vectors
vector1 = torch.tensor([4.0, 3.0, 2.0])
vector2 = torch.tensor([5.0, 1.0, 7.0])

diff_vector = vector1 - vector2
diff_vector_alt = torch.sub(vector1, vector2)

print("Difference of vectors:", diff_vector)

sum_vector = vector1 + vector2
sum_vector_alt = torch.add(vector1, vector2)

print("Sum of vectors:", sum_vector)

prod_vector = torch.mul(vector1, vector2)

print("Element-wise product of vectors:", prod_vector)

dot_product = torch.dot(vector1, vector2)

print("Dot product of vectors:", dot_product)


cross_product =  torch.linalg.cross(vector1, vector2)

print("Cross product of 3D vectors:", cross_product)