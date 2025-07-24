def fbGenerator(n):
    a, b = 0, 1
    for i in range(1, n):
        yield a
        a, b = b, a + b

for item in fbGenerator(8):
    print(item)