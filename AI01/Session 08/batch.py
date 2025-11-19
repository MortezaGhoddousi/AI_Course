import numpy as np
import matplotlib.pyplot as plt

W1 = np.zeros((2,2))
W2 = np.zeros((1,2))
b1 = np.zeros((2,1))
b2 = np.zeros((1,1))

p1 = np.array([[ 2],[ 4]])
p2 = np.array([[ 4],[ 2]])
p3 = np.array([[-2],[-2]])

P = np.hstack([p1, p2, p3]) 

T = np.array([[26, 26, -26]])

MSE = []

for i in range(100):
    # Forward pass
    n1 = W1 @ P + b1
    a1 = n1**2 

    n2 = W2 @ a1 + b2
    a2 = n2

    e = T - a2 
    mse = np.mean((e)**2)
    MSE.append(mse)
    Fdot2 = np.ones_like(n2)    
    Fdot1 = 2 * n1

    S2 = -2 * e * Fdot2
    S1 = (W2.T @ S2) * Fdot1

    dW2 = S2 @ a1.T
    db2 = np.sum(S2, axis=1, keepdims=True)

    dW1 = S1 @ P.T
    db1 = np.sum(S1, axis=1, keepdims=True)

    alpha = 0.01

    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1

print(W1, W2, b1, b2)

plt.figure()

plt.plot(MSE)

plt.show()