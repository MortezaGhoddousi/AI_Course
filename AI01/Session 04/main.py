# import matplotlib.pyplot as plt
# import numpy as np
# x = np.linspace(-2, 2, 50)
# y = np.ones(x.shape)*-0.5

# p1 = np.array([[-1, 1]])
# p2 = np.array([[-1, -1]])
# p3 = np.array([[0, 0]])
# p4 = np.array([[1, 0]])


# p5 = np.array([[-2, 0]])
# p6 = np.array([[1, 1]])
# p7 = np.array([[0, 1]])
# p8 = np.array([[-1, -2]])

# plt.figure()
# plt.plot(y, x)
# plt.scatter(p1[0][0], p1[0][1], marker='*', linewidths=2)
# plt.scatter(p2[0][0], p2[0][1], marker='*', linewidths=2)
# plt.scatter(p3[0][0], p3[0][1], marker='o', linewidths=2)
# plt.scatter(p4[0][0], p4[0][1], marker='o', linewidths=2)


# plt.scatter(p5[0][0], p5[0][1], linewidths=4)
# plt.scatter(p6[0][0], p6[0][1], linewidths=4)
# plt.scatter(p7[0][0], p7[0][1], linewidths=4)
# plt.scatter(p8[0][0], p8[0][1], linewidths=4)


# plt.xlim(-2, 2)
# plt.grid()
# plt.show()

# w = np.array([[1, 0]])
# b = 0.5

# def hardlim(n):
#     if n>=0:
#         return 0
#     else:
#         return 1
    
# n = np.matmul(w, np.transpose(p1))
# a1 = hardlim(n)

# n = np.matmul(w, np.transpose(p2))
# a2 = hardlim(n)

# n = np.matmul(w, np.transpose(p3))
# a3 = hardlim(n)

# n = np.matmul(w, np.transpose(p4))
# a4 = hardlim(n)


# print(a1, a2, a3, a4)


# n = np.matmul(w, np.transpose(p5))
# a5 = hardlim(n)

# n = np.matmul(w, np.transpose(p6))
# a6 = hardlim(n)

# n = np.matmul(w, np.transpose(p7))
# a7 = hardlim(n)

# n = np.matmul(w, np.transpose(p8))
# a8 = hardlim(n)


# print(a5, a6, a7, a8)


import numpy as np

def hardlims(n):
    if n[0] >= 0:
        a0 = 1
    else:
        a0 = -1
    if n[1] >= 0:
        a1 = 1
    else:
        a1 = -1
    return np.array([[a0], 
                     [a1]])


p1 = np.array([[.5],
               [-.5],
               [.5],
               [-.5]])

p2 = np.array([[.5],
               [.5],
               [-.5],
               [-.5]])

t1 = np.array([[1],
               [-1]])

t2 = np.array([[1],
               [1]])

w = np.zeros([2, 4])
b = np.zeros([2, 1])

for i in range(100):
    n = np.matmul(w, p1) + b
    a = hardlims(n)
    e = t1-a
    w = w+e*np.transpose(p1)

    n = np.matmul(w, p2) + b
    a = hardlims(n)
    e = t2-a
    w = w+e*np.transpose(p2)

print("weights:", w)
print("biases:",b)

n = np.matmul(w, p1) + b
a1 = hardlims(n)
n = np.matmul(w, p2) + b
a2 = hardlims(n)
print("test1:",a1)
print("test2",a2)


