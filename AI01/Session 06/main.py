import numpy as np

a = (np.exp(2)-np.exp(-2)) / (np.exp(2)+np.exp(-2))
print(a)

# p1 = np.array([[1],
#                [-1],
#                [-1]])

# p2 = np.array([[1],
#                [1],
#                [-1]])

# t1 = -1
# t2 = 1

# alpha = 0.2

# w = np.zeros([1, 3])
# b = 0

# for i in range(100):
#     a = np.matmul(w, p1) + b
#     e = t1 - a
#     w = w + 2*alpha*e*np.transpose(p1)
#     b = b + 2*alpha*e

#     a = np.matmul(w, p2) + b
#     e = t2 - a
#     w = w + 2*alpha*e*np.transpose(p2)
#     b = b + 2*alpha*e

# print(np.round(w))
# print(np.round(b))

# a = np.matmul(w, p1) + b
# print(a, t1)

# a = np.matmul(w, p2) + b
# print(a, t2)