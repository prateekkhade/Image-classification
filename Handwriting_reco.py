# Handwriting recognition in python using Tensorflow

#Importing libraries
import tensorflow as tf

#Importing datasets (MNIST)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

# Placeholder x for MNIST images (None for many images)
x = tf.placeholder(tf.float32, [None,784])

# Weights and biases inside variables
W = tf.Variable(tf.zeros([784,10])) #W and b both are zeros initially,
b = tf.Variable(tf.zeros([10])) #since it doesn't matter what they are as we're gonna train them

# Implementing the model (it takes only one line to implement it)
y = tf.nn.softmax(tf.matmul(x,W) + b)

# Adding placeholder for cross-entropy
y_ = tf.placeholder(tf.float32, ([None, 10]))

# Implementing the cross entropy function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ *tf.log(y), reduction_indices = [1]))

# Optimization algorithm (Gradient Descent in this case)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Operation to initialize the variables we created
init = tf.global_variables_initializer()

# Launch the model in a session
sess = tf.Session()
sess.run(init)

# Training the model
for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

# Does the prediction matches the truth?
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# Accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Accuracy on the test data
print(sess.run(accuracy, feed_dict = {x: mnist.test.images, y_:mnist.test.labels}))
