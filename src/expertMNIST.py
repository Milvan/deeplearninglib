import mlp
import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# x is the indput to the nn
x = tf.placeholder(tf.float32,  [None,784])

# y_ is the value y should be
y_ = tf.placeholder(tf.float32, [None, 10])

mlp = mlp.MLP(x, y_, [(784,500,tf.nn.sigmoid), (500, 10,tf.nn.softmax)]) #94.7%
#mlp = mlp.MLP(x, y_, [(784, 10,tf.nn.softmax)]) #92.2%
#mlp = mlp.MLP(x, y_, [(784,500,tf.nn.sigmoid), (500,250,tf.nn.sigmoid),(250, 10,tf.nn.softmax)]) #77%
#mlp = mlp.MLP(x, y_, [(784,900,tf.nn.sigmoid), (900, 10,tf.nn.softmax)]) #90.8%
#mlp = mlp.MLP(x, y_, [(784,1500,tf.nn.sigmoid), (1500,500,tf.nn.sigmoid),(500, 10,tf.nn.softmax)]) #


#Cross entropy for all data points, we whant to minimize this
#Cross entropy is basically a way of measuring the difference betweene two probability distrubutions, in our case, the predicted y values and the actual y_ values.
#cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#to train the nn, we will use a gradient decent optimizer uses the cross entropy gradient
#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#utility thing required if variables are used
init = tf.initialize_all_variables()

#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess = tf.Session()
sess.run(init)

for i in range(2000):
	batch_xs, batch_ys = mnist.train.next_batch(1000)
	sess.run(mlp.train_gd(0.001), feed_dict={x:batch_xs, y_:batch_ys})
	if i%31==0:
		#argmax returns the index of the largest value in the supplied vector in the given dimension.
		#for us this mean that the left value will be the predicted value of the nn and the right value will be the actual value.
		correct_prediction = tf.equal(tf.argmax(mlp.run(),1),tf.argmax(y_,1))

		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

		print "iteration ", i, (sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels}))

