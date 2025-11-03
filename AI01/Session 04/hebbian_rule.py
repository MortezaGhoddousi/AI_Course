import numpy as np

def hardlim(n):
    if n>=0:
        return 1
    else:
        return 0

x = np.linspace(-2, 2, 50)
y = np.ones(x.shape)*-0.5

p1 = np.array([[-1], 
               [1]])
p2 = np.array([[-1], 
               [-1]])
p3 = np.array([[0], 
               [0]])
p4 = np.array([[1], 
               [0]])

t1 = 1
t2 = 1
t3 = 0
t4 = 0


p5 = np.array([[-2], 
               [0]])
p6 = np.array([[1], 
               [1]])
p7 = np.array([[0], 
               [1]])
p8 = np.array([[-1], 
               [-2]])

w = np.random.random([1, 2])
b = np.random.random()

print(w, b)

for i in range(100):
    n = np.matmul(w, p1)
    a = hardlim(n)
    w = w + t1*np.transpose(p1)

    n = np.matmul(w, p2)
    a = hardlim(n)
    w = w + t2*np.transpose(p2)

    n = np.matmul(w, p3)
    a = hardlim(n)
    w = w + t3*np.transpose(p3)

    n = np.matmul(w, p4)
    a = hardlim(n)
    w = w + t4*np.transpose(p4)

print(w)

n = np.matmul(w, p5)
a5 = hardlim(n)

n = np.matmul(w, p6)
a6 = hardlim(n)

n = np.matmul(w, p7)
a7 = hardlim(n)

n = np.matmul(w, p8)
a8 = hardlim(n)

print(a5, a6, a7, a8)
