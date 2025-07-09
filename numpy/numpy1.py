import numpy as np
from PIL import Image

x = np.array([[[0, 1, 2], [2, 3, 7], [4, 6, 5]],
              [[2, 3, 5], [5, 1, 9], [6, 4, 8]],
              [[8, 1, 1], [4, 4, 3], [2, 0, 7]]])

x1 = np.unique(x)
print(f"sum of unique items of array: {x1.sum()}")

rand_array = np.random.randint(1, 10, (5, 6))
print(f"shape of random array: {rand_array.shape}")
print(f"random array: {rand_array}")

rand_array_unique = np.unique(rand_array)
print(f"unique elements of random array: {rand_array_unique}")

print(f"sum of unique items of random array: {rand_array_unique.sum()}")

print("Start rotating image")
image_path = "bird.jpg"
image = np.array(Image.open(image_path))

imageRotated = np.rot90(image)
imageRotated_K_3 = np.rot90(image, k=3)

Image.fromarray(imageRotated).save("new_image.jpg")
Image.fromarray(imageRotated_K_3).save("new_image_k_3.jpg")
print("Finish rotating image")