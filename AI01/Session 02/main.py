import numpy as np

def hardlim(n):
    if n >= 0:
        return 1
    else:
        return 0

p1 = np.array([[1],
               [-1],
               [-1]])

p2 = np.array([[1],
               [1],
               [-1]])

t1 = 0
t2 = 1

p1_test = np.array([[0.85],
                   [-0.65],
                   [-0.79]])

p2_test = np.array([[0.8],
                   [0.75],
                   [-0.7]])

w = np.array([[0, 0, 0]])
# w = np.zeros([1, 3])
b = 0

for i in range(2):
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

print(w)
print(b)


n = np.matmul(w, p1_test) + b
a = hardlim(n)

print(a, "p1 Test (0)")

n = np.matmul(w, p2_test) + b
a = hardlim(n)

print(a, "p2 Test (1)")
