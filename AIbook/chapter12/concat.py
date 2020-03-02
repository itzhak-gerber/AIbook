import numpy as np

a=np.array([[1,2,3],[4,5,6]])
b=np.array([[7,8,9],[10,11,12]])
print("a")
print(a)
print("b")
print(b)
c=np.concatenate([a,b],0)
print('c')
print(c)
d=np.concatenate([a,b],1)
print('d')
print(d)