import tensorflow as tf
import numpy as np

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


class MLP(object):
	def __init__(self, input_placeholder, labels_placeholder, layers_shape_and_activation):
		self.n_layers = len(layers_shape_and_activation)
		self.layers_shape_and_activation = layers_shape_and_activation
		self.input_placeholder = input_placeholder
		self.labels_placeholder = labels_placeholder
		# Wheights for the nn. These will be update every step of the training
		self.Weights = map(lambda (i,o,_a): weight_variable([i,o]), layers_shape_and_activation)
		# baias, spaceial wheigt
		self.biases = map(lambda (_i,o,_a): bias_variable([o]), layers_shape_and_activation)

	def run(self):
		activation = self.layers_shape_and_activation[0][2]
		y = activation(tf.matmul(self.input_placeholder,self.Weights[0])+self.biases[0])
		for i in range(1,self.n_layers):
			activation = self.layers_shape_and_activation[i][2]
			y = activation(tf.matmul(y,self.Weights[i])+self.biases[i])
		return y
	
	def train_gd(self, train_speed):
		cross_entropy = -tf.reduce_sum(self.labels_placeholder*tf.log(self.run()))
		train_step = tf.train.GradientDescentOptimizer(train_speed).minimize(cross_entropy)
		return train_step
		
