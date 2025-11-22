import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

try:
    df = pd.read_csv("data.csv")
except:
    df = pd.read_csv("AI01/Session 09/data.csv")

T = df['diagnosis'].to_numpy()
P = df.drop(columns=['id', 'diagnosis'], errors='ignore').to_numpy()

T = np.transpose(T)
P = np.transpose(P)

scaler = MinMaxScaler()
P = scaler.fit_transform(P)

P_train = P[:, :500]
P_test = P[:, 500:]

# Convert M/B to 1/0
for i, t in enumerate(T):
    if t == "M":
        T[i] = 1
    else:
        T[i] = 0

T_train = T[:500].astype(float)
T_test = T[500:].astype(float)
T = T.astype(float)

# ------------------------------------------------------------
# Neural network initialization
# ------------------------------------------------------------
number_of_neurons = 64
W1 = np.random.randn(number_of_neurons, 30) * 0.01
W2 = np.random.randn(1, number_of_neurons) * 0.01
b1 = np.zeros((number_of_neurons, 1), dtype=float)
b2 = np.zeros((1, 1), dtype=float)

MSE = []
ACC = []      # <----- STORE TRAIN ACCURACY
alpha = 0.001

# ------------------------------------------------------------
# Training loop
# ------------------------------------------------------------
for i in range(100):
    # Forward pass
    n1 = W1 @ P_train + b1
    a1 = n1

    n2 = W2 @ a1 + b2
    a2 = 1 / (1 + np.exp(-n2))

    # Error + MSE
    e = T_train - a2
    mse = np.mean(e**2)
    MSE.append(mse)

    # --------- Compute training accuracy for each iteration ----------
    mapped_data_train_iter = np.where(a2[0] >= 0.5, 1, 0)
    acc_train_iter = np.mean(T_train == mapped_data_train_iter)
    ACC.append(acc_train_iter)
    # -----------------------------------------------------------------

    # Derivatives
    Fdot2 = a2 * (1 - a2)
    Fdot1 = np.ones_like(n1)

    S2 = -2 * e * Fdot2
    S1 = (W2.T @ S2) * Fdot1

    # Gradients
    dW2 = S2 @ a1.T
    db2 = np.sum(S2, axis=1, keepdims=True)

    dW1 = S1 @ P_train.T
    db1 = np.sum(S1, axis=1, keepdims=True)

    # Update weights
    W2 -= alpha * dW2.astype(float)
    b2 -= alpha * db2.astype(float)
    W1 -= alpha * dW1.astype(float)
    b1 -= alpha * db1.astype(float)

# ------------------------------------------------------------
# Test results
# ------------------------------------------------------------
n1 = W1 @ P_test + b1
a1 = n1
n2 = W2 @ a1 + b2
a2_test = 1 / (1 + np.exp(-n2))

threshold = 0.5
mapped_data_test = np.where(a2_test[0] >= threshold, 1, 0)

print("MSE for test data:", np.mean((T_test - mapped_data_test)**2))
accuracy_test = np.mean(T_test == mapped_data_test)
print("Accuracy for test data:", accuracy_test)

# ------------------------------------------------------------
# Train results
# ------------------------------------------------------------
n1_train = W1 @ P_train + b1
a1_train = n1_train
n2_train = W2 @ a1_train + b2
a2_train = 1 / (1 + np.exp(-n2_train))

mapped_data_train = np.where(a2_train[0] >= threshold, 1, 0)

print("MSE for train data:", np.mean((T_train - mapped_data_train)**2))
accuracy_train = np.mean(T_train == mapped_data_train)
print("Accuracy for train data:", accuracy_train)

# ------------------------------------------------------------
# Plot training MSE and Accuracy
# ------------------------------------------------------------
plt.figure(figsize=(12,5))

# ---- Plot MSE ----
plt.subplot(1, 2, 1)
plt.plot(MSE)
plt.title("Mean Squared Error over Iterations")
plt.xlabel("Iterations")
plt.ylabel("MSE")

# ---- Plot Accuracy ----
plt.subplot(1, 2, 2)
plt.plot(ACC)
plt.title("Training Accuracy over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.show()
