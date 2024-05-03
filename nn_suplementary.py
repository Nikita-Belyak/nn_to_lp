from tensorflow.keras.models import Model
import networkx as nx
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

import os
import glob

def nn_train_model(layers, x_train, y_train, epochs=1000, batch_size=32):
    """
    Trains a neural network model with the given architecture and training data.

    Parameters:
    layers (list of int): The architecture of the neural network. Each integer in the list represents the number of neurons in a layer.
    x_train (numpy array or list): The training data.
    y_train (numpy array or list): The labels for the training data.
    epochs (int, optional): The number of epochs to train the model for. Default is 1000.
    batch_size (int, optional): The batch size to use when training the model. Default is 32.

    Returns:
    model (keras.Model): The trained model.

    This function does the following:
    1. Defines a neural network model with the given architecture. The model uses ReLU activation functions in all layers.
    2. Compiles the model with the Adam optimizer and mean squared error loss.
    3. Trains the model on the given training data.
    4. Checks if there are any files in the 'data/nn' directory and deletes them if there are.
    5. Gets the weights and biases from each layer of the model and saves them as .npy files in the 'data/nn' directory.
    6. Returns the trained model.
    """
    # Define the model
    model = Sequential()
    model.add(Dense(layers[0], activation='relu', input_shape=(1,)))

    for layer_size in layers[1:]:
        model.add(Dense(layer_size, activation='relu'))

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    # Check if there are any files in the data/nn directory
    files = glob.glob('data/nn/*')
    if files:
        # If there are, delete them
        for file in files:
            os.remove(file)

    # Get the weights and biases from each layer
    for i, layer in enumerate(model.layers):
        weights, biases = layer.get_weights()

        # Save the weights and biases as .npy files
        np.save(f'data/nn/layer_{i}_matrix_W.npy', weights)
        np.save(f'data/nn/layer_{i}_matrix_b.npy', biases)

    return model


def plot_neural_network(model, x_test):
    """
    Plots a neural network model's structure and activations.

    Parameters:
    model (keras.Model): The neural network model to plot.
    x_test (numpy array or list): The input data to use for the activations.

    This function does the following:
    1. Gets the output of the model for the given input and prints the first prediction.
    2. Gets the activations of each layer of the model for the given input.
    3. Prints the activations of each layer.
    4. Creates a directed graph where each node represents a neuron and each edge represents a connection between neurons.
    5. Defines the position of each node based on its layer and index within the layer.
    6. Defines the color of each node based on its activation (red for non-zero activations, blue for zero activations).
    7. Draws the graph with the defined positions and colors.
    8. Labels each node with its activation.
    9. Shows the plot.
    """
    # Get the output of the model for a given input
    y_test = model.predict(x_test)

    print(f'Prediction for {x_test[0]}: {y_test[0]}')

    # Get the activations of each layer
    activations = []
    for layer in model.layers:
        model_layer = Model(inputs=model.input, outputs=layer.output)
        activations.append(model_layer.predict(x_test))

    # Print the activations
    for i, activation in enumerate(activations):
        print(f'Layer {i + 1} activation:')
        print(activation)

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes for each neuron
    for i, activation in enumerate(activations):
        for j in range(len(activation[0])):
            G.add_node((i, j))

    # Add edges between neurons
    for i in range(len(activations) - 1):
        for j in range(len(activations[i][0])):
            for k in range(len(activations[i + 1][0])):
                G.add_edge((i, j), (i + 1, k))

    # Define the position of each node
    pos = {node: (node[0], -node[1]) for node in G.nodes()}

    # Define the color of each node
    color_map = []
    for node in G:
        # If the activation is non-zero, the color is red. Otherwise, the color is blue.
        color_map.append('red' if activations[node[0]][0][node[1]] != 0 else 'blue')

    # Draw the graph
    nx.draw(G, pos, node_color=color_map, with_labels=False)  # Turn off default labels

    # Label the nodes with their activations
    labels = {}
    for i, activation in enumerate(activations):
        for j in range(len(activation[0])):
            labels[(i, j)] = f'{activation[0][j]:.2f}'

    # Define label positions
    label_pos = {node: (pos[node][0], pos[node][1] + 0.1) for node in G.nodes()}

    nx.draw_networkx_labels(G, label_pos, labels=labels)

    plt.show()