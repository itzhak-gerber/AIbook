import numpy as np

x=np.array([[10],[7],[3]])
print("x:")
print(x)
print()
w1=np.zeros((2,3))
w1[0]=[-7,4,14]
w1[1]=[4,-6,3]
print("w1:")
print(w1)
print()

u=w1.dot(x)
print("u:")
print(u)
print()

b=np.array([-0.6,-0.3])
print("b:")
print(b)
print()

b=np.reshape(b,(1,2))
print("b:")
print(b)

b=b.T
print("b.T:")
print(b)
print()

z1=u+b
print("z1:")
print(z1)
print()

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

a1=sigmoid(z1)
print("a1:")
print(a1)
print()

w2=np.array([[4,5]])
print("w2:")
print(w2)
print()

u2=w2.dot(a1)
print("u2:")
print(u2)
print()

b2=np.array([[-6]])
print("b2:")
print(b2)
print()
z2=u2+b2
print("z2:")
print(z2)
print()

a2=sigmoid(z2)
print("a2:")
print(a2)

