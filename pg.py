import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns



l = [10,20,30,40,50,60,70,80,90,100]

arr = np.array([3,5,7,9,11])
arr2D = np.array([[1,2,3],[4,5,6],[7,8,9]])


print(arr)
print(type(arr))
print(type(l))
print(np.__version__)
print("-----------------")
print(arr2D.ndim)

print("-------------------")


for i, j in enumerate(l):
    print(f'{i} - {j}')



n = np.array(l)
print(n)
print(l)
print(type(n))


x = random.normal(loc=1, scale=2, size=(2,3))
print(x)
sns.distplot(random.normal(size=1000), hist=False)
plt.show()

# df = sns.load_dataset("penguins")
# sns.pairplot(df, hue="species")
# plt.show()



