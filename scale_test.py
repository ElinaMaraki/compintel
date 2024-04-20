import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Generate random data (2802 samples, each with 1000 features)
np.random.seed(42)  # Set random seed for reproducibility
data = np.random.rand(10, 3)  # Random data in the shape (2802, 1000)
print(data)
# Flatten the data into a 1D array (concatenate all rows)
flattened_data = data.ravel()
print(flattened_data.shape)
print(flattened_data)
# Create MinMaxScaler instance
scaler = MinMaxScaler(feature_range=(0, 1))

# Apply MinMaxScaler to the flattened data
scaled_data_flat = scaler.fit_transform(flattened_data.reshape(-1, 1))

# Reshape the scaled flattened data back to the original shape
scaled_data = scaled_data_flat.reshape(data.shape)

# Inverse transform to get back the original data
original_data_flat = scaler.inverse_transform(scaled_data.ravel().reshape(-1, 1))

# Reshape the original data back to its original shape
original_data = original_data_flat.reshape(data.shape)
print(original_data)
# Print the shape of scaled data (should be the same as input data shape)
print("Scaled Data Shape:", scaled_data.shape)

# Optionally, print the scaled data
print("Scaled Data:")
print(scaled_data)
