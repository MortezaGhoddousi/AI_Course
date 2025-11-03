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

for i in range(10000):
    n = np.matmul(w, p1) + b
    a = hardlim(n)
    e = t1-a
    w = w + e*np.transpose(p1)
    b = b + e 

    n = np.matmul(w, p2) + b
    a = hardlim(n)
    e = t2-a
    w = w + e*np.transpose(p2)
    b = b + e 

    n = np.matmul(w, p3) + b
    a = hardlim(n)
    e = t3-a
    w = w + e*np.transpose(p3)
    b = b + e 

    n = np.matmul(w, p4) + b
    a = hardlim(n)
    e = t4-a
    w = w + e*np.transpose(p4)
    b = b + e 

print(w)
print(b)

n = np.matmul(w, p5) + b
a5 = hardlim(n)

n = np.matmul(w, p6) + b
a6 = hardlim(n)

n = np.matmul(w, p7) + b
a7 = hardlim(n)

n = np.matmul(w, p8) + b
a8 = hardlim(n)

print(a5, a6, a7, a8)
