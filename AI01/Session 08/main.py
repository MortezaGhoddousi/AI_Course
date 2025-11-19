import numpy as np

alpha = 0.5
W1 = np.array([[1, -1],
               [1, 0]])

W2 = np.array([[2, 1]])

b1 = np.array([[1],
               [-1]])

b2 = np.array([[-1]])

p = np.array([[1],
              [1]])

t =  np.array([[2]])

n1 = np.matmul(W1, p) + b1

print(n1, "n1")
a1 = np.empty((0, 1))


for i in n1:
    val = i[0]**2
    a1 = np.append(a1, [[val]], axis=0)

    
print(a1, "a1")

a2 = np.matmul(W2, a1) + b2
print(a2, "a2")

e = t - a2
print(e, "e")

F1_dot = np.array([[0, 0],
                   [0, 0]])

F1_dot[0][0] = 2*n1[0][0]
F1_dot[1][1] = 2*n1[1][0]

print(F1_dot, "F1_dot")

F2_dot = np.array([[1]])

S2 = -2 * F2_dot * e
print(S2, "S2")

t = np.matmul(F1_dot, np.transpose(W2))
S1 = np.matmul(t, S2)
print(S1)

W1 = W1 - alpha * np.matmul(S1, np.transpose(p))
print(W1, "W1")

b1 = b1 - alpha * S1
print(b1, "b1")

W2 = W2 - alpha * np.matmul(S2, np.transpose(a1))
print(W2, "W2")

b2 = b2 - alpha * S2
print(b2, "b2")

