import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# -----------------------------------------------------
# 1. Load Data
# -----------------------------------------------------
try:
    df = pd.read_csv("data.csv")
except:
    df = pd.read_csv("AI01/Session 09/data.csv")

# Map diagnosis to binary  (M=1, B=0)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Extract Target (T) and Input (P)
T = df['diagnosis'].to_numpy().astype(float)
P = df.drop(columns=['id', 'diagnosis'], errors='ignore').to_numpy().astype(float)

T = T.reshape(-1, 1)   # make it (569, 1)

# -----------------------------------------------------
# 2. Normalize Input
# -----------------------------------------------------
scaler = MinMaxScaler()
P = scaler.fit_transform(P)

# -----------------------------------------------------
# 3. Train/Test Split
# -----------------------------------------------------
P_train, P_test, T_train, T_test = train_test_split(
    P, T, test_size=0.15, random_state=42
)

# -----------------------------------------------------
# 4. Build 2-Layer Neural Network
# -----------------------------------------------------
model = Sequential([
    Dense(64, activation='linear', input_shape=(30,)),   # purelin
    Dense(1, activation='sigmoid')                       # logsig
])

# -----------------------------------------------------
# 5. Compile Model
# -----------------------------------------------------
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.MeanSquaredError()]
)

# -----------------------------------------------------
# 6. Train Network
# -----------------------------------------------------
history = model.fit(
    P_train, T_train,
    epochs=100,
    batch_size=64,
    validation_split=0.15,
    verbose=1
)

# -----------------------------------------------------
# 7. Evaluate on Test Set
# -----------------------------------------------------
loss, accuracy, mse = model.evaluate(P_test, T_test, verbose=0)
print("\nTest Loss:", loss)
print("Test Accuracy:", accuracy)
print("Test MSE:", mse)

# -----------------------------------------------------
# 8. Plot MSE and Accuracy
# -----------------------------------------------------
plt.figure(figsize=(12,5))

# ---- MSE Plot ----
plt.subplot(1, 2, 1)
plt.plot(history.history['mean_squared_error'], label='Train MSE')
plt.plot(history.history['val_mean_squared_error'], label='Validation MSE')
plt.title('MSE During Training')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()

# ---- Accuracy Plot ----
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy During Training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# -----------------------------------------------------
# 9. Print Model Summary
# -----------------------------------------------------
model.summary()
