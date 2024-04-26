import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import constraints

# Define the function
def f(x):
    return x**2 + 3*np.sin(5*x)


# Train a neural network to approximate the function f(x) = x^2 + 3sin(5x)
# Generate training data
x_train = np.linspace(-4, 4, 1000)
y_train = f(x_train)

# Define the model
model = Sequential([
    Dense(8, activation='relu', input_shape=(1,)),
    Dense(16, activation='relu'), 
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(x_train, y_train, epochs=1000, batch_size=32)

# Get the weights and biases from each layer
for i, layer in enumerate(model.layers):
    weights, biases = layer.get_weights()
    
    # Save the weights and biases as .npy files
    np.save(f'data/nn/layer_{i}_matrix_W.npy', weights)
    np.save(f'data/nn/layer_{i}_matrix_b.npy', biases)

# create a function that replicates a trained ICNN model 

import numpy as np
import glob
import os

# Get the list of all files in the folder
files = glob.glob('./data/icnn/layer*_matrix_*.npy')

# Sort the files
files.sort()


# Initialize the dictionaries to hold the matrices
A_matrices = {}
b_vectors = {}
W_matrices = {}


# Load the matrices from the files
for file in files:
    # Extract the filename from the file path
    filename = os.path.basename(file)

    # Extract the layer number and the matrix name from the filename
    parts = filename.split('_')
    layer_number = int(parts[1])  # Extract the number after 'layer'

    matrix_name = parts[3]  # Extract the matrix name
    print(matrix_name)


    # Load the matrix from the file
    matrix = np.load(file)

    # Store the matrix in the appropriate dictionary
    if matrix_name == 'A.npy':
        A_matrices[layer_number] = matrix
    elif matrix_name == 'b.npy':
        b_vectors[layer_number] = matrix
    elif matrix_name == 'W.npy':
        W_matrices[layer_number] = matrix



# Define the function to evaluate the ICNN model
def evaluate_icnn(x, A_matrices, b_vectors, W_matrices):
    # Initialize the output
    y = np.zeros_like(x)

    # Iterate over the layers
    for i in range(len(A_matrices)):
        print(i)
        # Get the matrices for the current layer
        A = A_matrices[i]
        b = b_vectors[i]
        #print(A)
        #print(b)

        # Compute the output of the current layer
        if i == 0:
            y = np.maximum(0, np.matmul(x, A) + b)
        else:
            W = W_matrices[i]
            print("Size of W:", W.shape)  # Print the size of W
            y = np.maximum(0, np.matmul(x, A) + b + np.matmul(y, W))

        print(y)

    return y[0][0]


# Generate test data
x_test = np.linspace(-4, 4, 100)

# Make predictions
y_pred_nn = model.predict(x_test)

# Compute the ICNN predictions
y_pred_icnn = [ evaluate_icnn(np.array([x]), A_matrices, b_vectors, W_matrices) for x in x_test]



import matplotlib.pyplot as plt
# Plot the function and the predictions
plt.figure(figsize=(10, 5))
plt.plot(x_test, f(x_test), label='True function')
plt.plot(x_test, y_pred_nn, label='NN Predicted function')
plt.plot(x_test, y_pred_icnn, label='ICNN Predicted function')
plt.legend()
plt.show()



