import numpy as np

# Fake activations from previous layer
X = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
])

# Step 1: Mean
mean = np.mean(X, axis=0)

# Step 2: Variance
variance = np.var(X, axis=0)

# Step 3: Normalize
epsilon = 1e-5

X_normalized = (X - mean) / np.sqrt(variance + epsilon)

# Step 4: Learnable parameters
gamma = np.ones(X.shape[1])
beta = np.zeros(X.shape[1])

# Step 5: Scale and shift
out = gamma * X_normalized + beta

print("Mean:")
print(mean)

print("\nVariance:")
print(variance)

print("\nNormalized Output:")
print(X_normalized)

print("\nFinal BatchNorm Output:")
print(out)