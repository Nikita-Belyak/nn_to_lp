
import argparse
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.linalg import sqrtm
import random

import sklearn
import matplotlib.pylab as pl
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss

# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,3"  # specify which GPU(s) to be used

parser = argparse.ArgumentParser()
parser.add_argument('--DATASET_X', type=str, default='styblinski_tang', help='which dataset to use for X')

parser.add_argument('--LAMBDA', type=float, default=1, help='Regularization constant for positive weight constraints')

parser.add_argument('--NUM_NEURON', type=int, default=16, help='number of neurons per layer')

parser.add_argument('--NUM_LAYERS', type=int, default=3, help='number of hidden layers before output')

parser.add_argument('--LR', type=float, default=1e-3, help='learning rate')

parser.add_argument('--ITERS', type=int, default=1000, help='number of iterations of training')

parser.add_argument('--BATCH_SIZE', type=int, default=1024, help='size of the batches')

parser.add_argument('--N_TEST', type=int, default=2048, help='number of test samples')

parser.add_argument('--INPUT_DIM', type=int, default=2, help='dimensionality of the input x')

opt = parser.parse_args()
print(opt)

def main():

	# specify the convex function class
	print("specify the convex function class")
	
	# save the number of neurons per layer
	hidden_size_list = [opt.NUM_NEURON for i in range(opt.NUM_LAYERS)]

	# add the output layer
	hidden_size_list.append(1)
	print(hidden_size_list)

	# create the neural network structure
	fn_model = Kantorovich_Potential(opt.INPUT_DIM, hidden_size_list)  

	# Define the test set
	print ("Define the test set")
	data_test = next(sample_data_gen(opt.DATASET_X, opt.N_TEST))

	saver = tf.train.Saver()
	
	# Running the optimization
	with tf.Session() as sess:

		compute_OT = ComputeOT(sess, opt.INPUT_DIM, fn_model,  opt.LR) # initilizing
		
		compute_OT.learn(opt.BATCH_SIZE, opt.ITERS, opt.DATASET_X, opt) # learning the optimal map
		
		for nl in range(0, len(hidden_size_list)-1):
				
				if nl == 0:
					save_A = fn_model.A[nl].eval()
					file_name_A = f"data/layer{nl}_matrix_A.npy"
					np.save(file_name_A, save_A)

					save_b = fn_model.b[nl].eval()
					file_name_b = f"data/layer{nl}_matrix_b.npy"
					np.save(file_name_b, save_b)

				else: 
					save_W = fn_model.W[nl].eval()
					file_name_W = f"data/layer{nl}_matrix_W.npy"
					#print(fn_model.W[nl].eval())
					np.save(file_name_W, save_W)

					save_A = fn_model.A[nl].eval()
					file_name_A = f"data/layer{nl}_matrix_A.npy"
					np.save(file_name_A, save_A)

					save_b = fn_model.b[nl].eval()
					file_name_b = f"data/layer{nl}_matrix_b.npy"
					np.save(file_name_b, save_b)
	

		
class ComputeOT:
	
	def __init__(self, sess, input_dim, f_model, lr):
		
		self.sess = sess 
		
		# use predefined structure of the neural network
		self.f_model = f_model

		# use the predefined dimentions of the input data 
		self.input_dim = input_dim
		
		# define the input and output placeholders
		self.x = tf.placeholder(tf.float32, [None, input_dim])
		self.y = tf.placeholder(tf.float32, [None])

		# define the forward pass
		self.fx = self.f_model.forward(self.x)

		# define the gradient of the forward pass
		[self.grad_fx] = tf.gradients(self.fx,self.x)
		
		# define the loss function
		self.f_loss = tf.reduce_mean(tf.square(self.fx - self.y))

		# define the optimizer for the loss function (adam optimizer is used here with learning rate = lr)
		self.f_optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(self.f_loss, var_list= self.f_model.var_list)
		
		# initialize the variables
		self.init = tf.global_variables_initializer()

	def learn(self, batch_size, iters, dataset_x, opt):
	 
		# Initialize the variables
		self.sess.run(self.init)

		# Generate the training data
		data_gen = sample_data_gen(dataset_x, batch_size)
		print ("training data is created")

		# generate training set 
		trainable_f_list = [self.f_optimizer , self.f_model.proj]
		
		for iteration in range(iters):

			x_train, y_train = next(data_gen)

			#Training the f_neural network
			_ = self.sess.run( trainable_f_list, feed_dict={self.x: x_train, self.y: y_train})

			f_loss = self.sess.run(self.f_loss, feed_dict={self.x: x_train, self.y: y_train}) 
			print ("Iterations = %i, f_loss = %.4f" %(iteration,f_loss))
					

class Kantorovich_Potential:
	''' 
		Modelling the Kantorovich potential as Input convex neural network (ICNN)
		input: y
		output: z = h_L
		Architecture: h_1     = ReLU^2(A_0 y + b_0)
					  h_{l+1} =   ReLU(A_l y + b_l + W_{l-1} h_l)
		Constraint: W_l > 0
	'''
	def __init__(self,input_size, hidden_size_list):

		# hidden_size_list always contains 1 in the end because it's a scalar output
		self.input_size = input_size
		self.num_hidden_layers = len(hidden_size_list)
		
		
		# list of matrices that interacts with input
		self.A = []
		for k in range(0, self.num_hidden_layers):
			self.A.append(tf.Variable(tf.random_uniform([self.input_size, hidden_size_list[k]], maxval=0.1), dtype=tf.float32))

		# list of bias vectors at each hidden layer 
		self.b = []
		for k in range(0, self.num_hidden_layers):
			self.b.append(tf.Variable(tf.zeros([1, hidden_size_list[k]]),dtype=tf.float32))

		# list of matrices between consecutive layers
		self.W = []
		for k in range(1, self.num_hidden_layers):
			self.W.append(tf.Variable(tf.random_uniform([hidden_size_list[k-1], hidden_size_list[k]], maxval=0.1), dtype=tf.float32))
		
		self.var_list = self.A +  self.b + self.W

		self.proj = [w.assign(tf.nn.relu(w)) for w in self.W]  # ensuring the weights to stay positive
		self.apply_projection = tf.group(*self.proj)  # Group the assignment operations

	def forward(self, input_y):
		
		# Using ReLU Squared
		z = tf.nn.leaky_relu(tf.matmul(input_y, self.A[0]) + self.b[0], alpha=0.2)
		z = tf.multiply(z,z)

		# # If we want to use ReLU and softplus for the input layer
		# z = tf.matmul(input_y, self.A[0]) + self.b[0]
		# z = tf.multiply(tf.nn.relu(z),tf.nn.softplus(z))
		
		# If we want to use the exponential layer for the input layer
		## z=tf.nn.softplus(tf.matmul(input_y, self.A[0]) + self.b[0])
		
		for k in range(1,self.num_hidden_layers):
			
			z = tf.nn.leaky_relu(tf.matmul(input_y, self.A[k]) + self.b[k] + tf.matmul(z, self.W[k-1]))

		return z  