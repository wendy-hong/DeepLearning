import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

tf.reset_default_graph()
# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784])  # mnist data size 28*28=784
y = tf.placeholder(tf.float32, [None, 10])  # 0-9 => 10 classes

# Set model weights
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Build model
pred = tf.nn.softmax(tf.matmul(x, W) + b)
# Minimize error using cross entropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))

# Parameter setting
learning_rate = 0.01
# Gradient Descent Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

training_epochs = 25
batch_size = 100
display_step = 1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)  # iterations

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, loss], feed_dict={x: batch_xs, y: batch_ys})
            avg_cost += c / total_batch

        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(avg_cost))
    print("Finished!")

    # Testing
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))  # dtype = bool
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
