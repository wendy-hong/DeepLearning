import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Parameter Setting
learning_rate = 0.001
training_epochs = 25
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256  # number of features in 1st layer
n_hidden_2 = 256  # number of features in 2nd layer
n_input = 784  # MNIST img shape: 28*28
n_classes = 10  # 0-9 classes

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

weights = {'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
           'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
           'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))}
biases = {'b1': tf.Variable(tf.random_normal([n_hidden_1])),
          'b2': tf.Variable(tf.random_normal([n_hidden_2])),
          'out': tf.Variable(tf.random_normal([n_classes]))}


def multilayer_perceptron(x, weights, biases):
    layer1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer1 = tf.nn.relu(layer1)
    layer2 = tf.add(tf.matmul(layer1, weights['h2']), biases['b2'])
    layer2 = tf.nn.relu(layer2)
    out_layer = tf.matmul(layer2, weights['out']) + biases['out']
    return out_layer


pred = multilayer_perceptron(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = mnist.train.num_examples // batch_size

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch

        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
    print("Finished!")

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))  # dtype = bool
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
