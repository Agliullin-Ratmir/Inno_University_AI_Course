import numpy as np

vector1 = np.array([4.0, 3.0, 2.0])
vector2 = np.array([5.0, 1.0, 7.0])


sum_vector = vector1 + vector2

sum_vector_alt = np.add(vector1, vector2)

print("Sum of vectors:", sum_vector)

diff_vector = vector1 - vector2
diff_vector_alt = np.subtract(vector1, vector2)

print("Difference of vectors:", diff_vector)

prod_vector_alt = np.multiply(vector1, vector2)

print("Element-wise product of vectors:", prod_vector)

dot_product = np.dot(vector1, vector2)

print("Dot product of vectors:", dot_product)

cross_product = np.cross(vector1, vector2)

print("Cross product of 3D vectors:", cross_product)