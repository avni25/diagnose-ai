import numpy as np


l = [10,20,30,40,50,60,70,80,90,100]

for i, j in enumerate(l):
    print(f'{i} - {j}')


n = np.array(l)

print(n)
print(l)
print(type(n))




