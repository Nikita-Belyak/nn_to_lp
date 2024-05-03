import numpy as np
import icnn_suplementary 
import nn_suplementary
import icnn_one_dim


# Define the function
def f(x):
    return x**2 + 3*np.sin(5*x)

#----------------nn-----------------
# Train a neural network to approximate the function f(x) = x^2 + 3sin(5x)
# Generate training data
x_train = np.linspace(-4, 4, 1000)
y_train = f(x_train)

# Train the model
model = nn_suplementary.nn_train_model([4, 4, 1], x_train, y_train)

# Get the output of the model for a given input
x_test = np.array([2.5])  
nn_suplementary.plot_neural_network(model, x_test)

#---------------icnn----------------

opt = icnn_one_dim.Options()
print(opt)

# Update the options
opt.update(NUM_NEURON=[4, 4])
print(opt)

icnn_one_dim.py.icnn_train(opt)

# Load the matrices from the files
A_matrices, b_vectors, W_matrices = icnn_suplementary.icnn_load_matrices(path='./data/icnn/layer*_matrix_*.npy')

icnn_suplementary.evaluate_icnn(x_test, A_matrices, b_vectors, W_matrices, True)



#-----------------plot----------------

# Generate test data
x_test = np.linspace(-4, 4, 100)

# Make predictions
y_pred_nn = model.predict(x_test)

# Compute the ICNN predictions
y_pred_icnn = [ icnn_suplementary.evaluate_icnn(np.array([x]), A_matrices, b_vectors, W_matrices) for x in x_test]

import matplotlib.pyplot as plt
# Plot the function and the predictions
plt.figure(figsize=(10, 5))
plt.plot(x_test, f(x_test), label='True function')
plt.plot(x_test, y_pred_nn, label='NN Predicted function')
plt.plot(x_test, y_pred_icnn, label='ICNN Predicted function')
plt.legend()
plt.show()



