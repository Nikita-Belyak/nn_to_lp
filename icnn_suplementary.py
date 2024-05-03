import glob
import os
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt


def icnn_load_matrices(path='./data/icnn/layer*_matrix_*.npy'):
    # Get the list of all files in the folder
    files = glob.glob(path)

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

        # Load the matrix from the file
        matrix = np.load(file)

        # Store the matrix in the appropriate dictionary
        if matrix_name == 'A.npy':
            A_matrices[layer_number] = matrix
        elif matrix_name == 'b.npy':
            b_vectors[layer_number] = matrix
        elif matrix_name == 'W.npy':
            W_matrices[layer_number] = matrix

    return A_matrices, b_vectors, W_matrices



def evaluate_icnn(x, A_matrices, b_vectors, W_matrices, verbose=False):
    
    # Initialize the output
    y = np.zeros_like(x)
    
    # List to store activations
    activations = []

    # Iterate over the layers
    for i in range(len(A_matrices)):
        #print("i = ", i)
        # Get the matrices for the current layer
        A = A_matrices[i]
        b = b_vectors[i]

        #print("A = ", A)
        #print("b = ", b)
        # Compute the output of the current layer
        if i == 0:
            y = np.maximum(0, np.matmul(x, A) + b)
        else:
            W = W_matrices[i]
            y = np.maximum(0, np.matmul(x, A) + b + np.matmul(y, W))

        # Store the activations
        activations.append(y)
    if verbose:
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
        label_pos = {node: (pos[node][0], pos[node][1] + 0.5) for node in G.nodes()}

        nx.draw_networkx_labels(G, label_pos, labels=labels)

        # Plot the output
        
        plt.plot(x, y)
        plt.show()

    return y[0][0]