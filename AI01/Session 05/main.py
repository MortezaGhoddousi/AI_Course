import numpy as np
import matplotlib.pyplot as plt

def harlims(n):
    a = np.array([[]])
    for i in n:
        if i>=0:
            a = np.append(a, 1)
        else:
            a = np.append(a, -1)
    return np.transpose(a)

def display_binary_image(pT):
    binary_p_image = (pT == -1).astype(int)
    binary_p_image = binary_p_image.reshape(6, 5)
    return binary_p_image

def add_salt_pepper_noise(vec, noise_ratio=0.1):
    noisy_vec = vec.copy()
    n = len(vec)
    num_noisy = int(noise_ratio*n)
    indices = np.random.choice(n, num_noisy, replace=False)
    noisy_vec[indices] *= -1
    return noisy_vec


p0T = np.array([[-1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1,-1, 1, 1, 1, -1]])
p1T = np.array([[-1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1]])
p2T = np.array([[1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1]])

p0 = np.transpose(p0T)
p1 = np.transpose(p1T)
p2 = np.transpose(p2T)

w = np.matmul(p0, p0T)+np.matmul(p1, p1T)+np.matmul(p2, p2T)

test = np.array([[4],
                 [5],
                 [-9],
                 [0]])



n = np.matmul(w, p0)
a0 = harlims(n)

n = np.matmul(w, p1)
a1 = harlims(n)

n = np.matmul(w, p2)
a2 = harlims(n)

binary_p0_image = display_binary_image(p0T)
binary_a0_image = display_binary_image(a0)

binary_p1_image = display_binary_image(p1T)
binary_a1_image = display_binary_image(a1)

binary_p2_image = display_binary_image(p2T)
binary_a2_image = display_binary_image(a2)

plt.figure()
plt.subplot(2, 3, 1)
plt.imshow(binary_p0_image, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.subplot(2, 3, 4)
plt.imshow(binary_a0_image, cmap='gray', interpolation='nearest')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(binary_p1_image, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.subplot(2, 3, 5)
plt.imshow(binary_a1_image, cmap='gray', interpolation='nearest')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(binary_p2_image, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.subplot(2, 3, 6)
plt.imshow(binary_a2_image, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.show()

# test 50%
p0T50 = np.array([[-1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,-1, -1, -1, -1, -1]])
p1T50 = np.array([[-1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
p2T50 = np.array([[1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

p050 = np.transpose(p0T50)
p150 = np.transpose(p1T50)
p250 = np.transpose(p2T50)

n = np.matmul(w, p050)
a050 = harlims(n)

n = np.matmul(w, p150)
a150 = harlims(n)

n = np.matmul(w, p250)
a250 = harlims(n)

binary_p050_image = display_binary_image(p0T50)
binary_a050_image = display_binary_image(a050)

binary_p250_image = display_binary_image(p2T50)
binary_a250_image = display_binary_image(a250)

binary_p150_image = display_binary_image(p1T50)
binary_a150_image = display_binary_image(a150)

plt.figure()
plt.subplot(2, 3, 1)
plt.imshow(binary_p050_image, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.subplot(2, 3, 4)
plt.imshow(binary_a050_image, cmap='gray', interpolation='nearest')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(binary_p150_image, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.subplot(2, 3, 5)
plt.imshow(binary_a150_image, cmap='gray', interpolation='nearest')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(binary_p250_image, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.subplot(2, 3, 6)
plt.imshow(binary_a250_image, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.show()


# test 67%
p0T67 = np.array([[-1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,-1, -1, -1, -1, -1]])
p1T67 = np.array([[-1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
p2T67 = np.array([[1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

p067 = np.transpose(p0T67)
p167 = np.transpose(p1T67)
p267 = np.transpose(p2T67)

n = np.matmul(w, p067)
a067 = harlims(n)

n = np.matmul(w, p167)
a167 = harlims(n)

n = np.matmul(w, p267)
a267 = harlims(n)

binary_p067_image = display_binary_image(p0T67)
binary_a067_image = display_binary_image(a067)

binary_p267_image = display_binary_image(p2T67)
binary_a267_image = display_binary_image(a267)

binary_p167_image = display_binary_image(p1T67)
binary_a167_image = display_binary_image(a167)

plt.figure()
plt.subplot(2, 3, 1)
plt.imshow(binary_p067_image, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.subplot(2, 3, 4)
plt.imshow(binary_a067_image, cmap='gray', interpolation='nearest')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(binary_p167_image, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.subplot(2, 3, 5)
plt.imshow(binary_a167_image, cmap='gray', interpolation='nearest')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(binary_p267_image, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.subplot(2, 3, 6)
plt.imshow(binary_a267_image, cmap='gray', interpolation='nearest')
plt.axis('off')

plt.show()


# test noisy%
p0_noisyT = add_salt_pepper_noise(p0T[0], noise_ratio=0.3)
p1_noisyT = add_salt_pepper_noise(p1T[0], noise_ratio=0.3)
p2_noisyT = add_salt_pepper_noise(p2T[0], noise_ratio=0.3)

p0_noisy = np.transpose(p0_noisyT)
p1_noisy = np.transpose(p1_noisyT)
p2_noisy = np.transpose(p2_noisyT)

n = np.matmul(w, p0_noisy)
a0_noisy = harlims(n)

n = np.matmul(w, p1_noisy)
a1_noisy = harlims(n)

n = np.matmul(w, p2_noisy)
a2_noisy = harlims(n)

binary_p0_noisy_image = display_binary_image(p0_noisyT)
binary_a0_noisy_image = display_binary_image(a0_noisy)

binary_p1_noisy_image = display_binary_image(p1_noisyT)
binary_a1_noisy_image = display_binary_image(a1_noisy)

binary_p2_noisy_image = display_binary_image(p2_noisyT)
binary_a2_noisy_image = display_binary_image(a2_noisy)

plt.figure()
plt.subplot(2, 3, 1)
plt.imshow(binary_p0_noisy_image, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.title("noisy digit")
plt.subplot(2, 3, 4)
plt.imshow(binary_a0_noisy_image, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.title("predicted digit")


plt.subplot(2, 3, 2)
plt.imshow(binary_p1_noisy_image, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.title("noisy digit")

plt.subplot(2, 3, 5)
plt.imshow(binary_a1_noisy_image, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.title("predicted digit")


plt.subplot(2, 3, 3)
plt.imshow(binary_p2_noisy_image, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.title("noisy digit")

plt.subplot(2, 3, 6)
plt.imshow(binary_a2_noisy_image, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.title("predicted digit")


plt.show()