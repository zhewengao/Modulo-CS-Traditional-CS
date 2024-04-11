import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
sys.path.append('D:\anaconda\lib\site-packages')
import cvxpy as cp
from scipy.optimize import minimize
##################################################################################################################
###################################################################################################################
#Task One
# Function to read MNIST data from CSV filey
def read_mnist_csv(filename):
    return pd.read_csv(filename, header=None).values[:, 1:]  # Exclude the label column

# Path to the MNIST dataset
data_path = "E:\\data compression\\project\\mnist_dataset (1)"

# Read MNIST train and test images from CSV files
train_images = read_mnist_csv(os.path.join(data_path, 'mnist_train.csv'))
test_images = read_mnist_csv(os.path.join(data_path, 'mnist_test.csv'))

# Sparsify data: Represent foreground with nonzero entries and background with zeros
train_images[train_images > 0.5] = 1
train_images[train_images <= 0.5] = 0
train_images = train_images.T

# Generate fat random matrices with varying numbers of columns M
def generate_random_matrix(n_rows, n_cols):
    matrix = np.random.randn(n_rows, n_cols)
    matrix /= np.linalg.norm(matrix, axis=0)
    return matrix

# Number of columns in the random matrix M
M_values = [50,100, 150, 200,300,500]
input_to_algorithm = {} #
input_to_algorithmshow = {}

for M in M_values:
    random_matrix = generate_random_matrix(M, 784)
    projected_data = np.dot(random_matrix, train_images)
    input_to_algorithm[M] = projected_data
    input_to_algorithmshow[M] = projected_data.T

# Plot some examples
num_examples = 1
for i in range(num_examples):
    plt.figure(figsize=(15, 3))
    for j, M in enumerate(M_values):
        plt.subplot(1, len(M_values), j + 1)
        plt.imshow(input_to_algorithmshow[M][i].reshape(-1, 1), cmap='gray', aspect='auto')
        plt.title(f'M = {M}')
        plt.axis('off')
    plt.show()

#######################################################################################################################
#####################################################################################################################
#Task two
# Modulo Compressed Sensing Recovery Algorithm
# def modulo_cs_recovery(y, A):
#     """Modulo Compressed Sensing Recovery Algorithm."""
#     x_hat = np.zeros(A.shape[1])
#     for i in range(A.shape[0]):
#         residue = y[i] - (np.dot(A[i], x_hat) - np.floor(np.dot(A[i], x_hat)))
#         x_hat += residue * A[i]
#     return x_hat


# Modulo Compressed Sensing Recovery Algorithm using optimization (P0)
def modulo_cs_recovery_optimized(y, A):
    m, n = A.shape

    # Define variables
    x_plus = cp.Variable(n)
    x_minus = cp.Variable(n)
    v = cp.Variable(m, integer=True)  # integer=True indicates that v is an integer variable

    # Define the objective function for l1-norm minimization
    objective = cp.Minimize(cp.sum(x_plus) + cp.sum(x_minus))

    result = cp.hstack([A, -A, -np.eye(m)]) @ cp.hstack([x_plus, x_minus, v]),

    # Define the constraint function
    constraints = [
        result == y,
        v >= 0,
        x_plus >= 0,
        x_minus >= 0
    ]



    # Define the optimization problem
    problem = cp.Problem(objective, constraints)

    # Solve the optimization problem using GLPK solver
    problem.solve(solver=cp.GLPK)

    # Extract the reconstructed signal
    x_hat = x_plus.value - x_minus.value

    return x_hat


# reconstruction_errors_modulo = []
# reconstruction_errors_optimized = []
#
# for M in M_values:
#     random_matrix = generate_random_matrix(784, M)
#
#     # Modulo Compressed Sensing Recovery
#     y_modulo = np.dot(train_images, random_matrix) - np.floor(np.dot(train_images, random_matrix))
#     print(y_modulo.shape)
#     x_hat_modulo = modulo_cs_recovery(y_modulo, random_matrix)
#
#     # Modulo Compressed Sensing Recovery using optimization
#     ax=np.dot(train_images, random_matrix).T
#     y_optimized = ax - np.floor(ax)
#     print(y_optimized.shape)
#
#     # Calculate Reconstruction Error
#     reconstructed_image_modulo = np.dot(x_hat_modulo, random_matrix.T)
#     # reconstructed_image_optimized = np.dot(x_hat_optimized, random_matrix.T)
#
#     error_modulo = np.mean((train_images - reconstructed_image_modulo) ** 2)
#     # error_optimized = np.mean((train_images - reconstructed_image_optimized) ** 2)
#
#     reconstruction_errors_modulo.append(error_modulo)
#     # reconstruction_errors_optimized.append(error_optimized)


# Reconstruct the signals
reconstructed_signals = {}
for M, y_optimized in input_to_algorithm.items():
    x_hat_optimized_list = []

    for i in range(y_optimized.shape[1]):  # 遍历每一列
        y_single = y_optimized[:, i] - np.floor(y_optimized[:, i])
        print(y_single.shape)
        x_hat_optimized = modulo_cs_recovery_optimized(y_single, random_matrix)
        x_hat_optimized_list.append(x_hat_optimized)

    # Convert the list to numpy array
    x_hat_optimized = np.array(x_hat_optimized_list)
    reconstructed_signals[M] = x_hat_optimized

# Print the shape of reconstructed signals for each M
for M, x_hat in reconstructed_signals.items():
    print(f"M = {M}, Reconstructed Signal Shape: {x_hat.shape}")

# Plotting Reconstruction Error
plt.figure()
plt.plot(M_values, reconstruction_errors_modulo, '-o', label='Modulo CS')
# plt.plot(M_values, reconstruction_errors_optimized, '-x', label='Modulo CS Optimized')
plt.xlabel('Number of Measurements (M)')
plt.ylabel('Reconstruction Error')
plt.title('Reconstruction Error vs Number of Measurements')
plt.legend()
plt.grid(True)
plt.show()