import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle


def generate(sample_size, classNum, diff, regression=False):
    np.random.seed(10)
    mean = np.random.randn(2)
    cov = np.eye(2)
    samples_per_class = sample_size // classNum
    X0 = np.random.multivariate_normal(mean, cov, samples_per_class)
    Y0 = np.zeros(samples_per_class)

    for ci, d in enumerate(diff):
        X1 = np.random.multivariate_normal(mean + d, cov, samples_per_class)
        Y1 = (ci + 1) * np.ones(samples_per_class)
        X0 = np.concatenate((X0, X1))
        Y0 = np.concatenate((Y0, Y1))

    if regression == False:  # one-hot
        class_ind = [Y0 == i for i in range(classNum)]
        Y0 = np.stack(class_ind, 1)
        Y0 = np.asarray(Y0, dtype=np.float32)
    X, Y = shuffle(X0, Y0)
    return X, Y


np.random.seed(10)
input_dim = 2
num_classes = 4
n_input = 2
n_label = 1
n_hidden = 200
learning_rate = 0.01
reg = 0.01
Epoches = 3000

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_label])
weights = {'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden], stddev=0.1)),
           'h2': tf.Variable(tf.random_normal([n_hidden, n_label], stddev=0.1))}
biases = {'h1': tf.Variable(tf.zeros([n_hidden])),
          'h2': tf.Variable(tf.zeros([n_label]))}

layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['h1']))
keep_prob = tf.placeholder("float")
layer_1_drop = tf.nn.dropout(layer_1, keep_prob)
layer2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['h2'])
y_pred = tf.maximum(layer2, 0.01 * layer2)  # Leaku ReLU

loss = tf.reduce_mean((y_pred - y) ** 2) + tf.nn.l2_loss(weights['h1']) * reg + tf.nn.l2_loss(weights['h2']) * reg
global_step = tf.Variable(0, trainable=False)
decaylearning_rate = tf.train.exponential_decay(learning_rate, global_step, 1000, 0.9)
train = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(Epoches):
        X, Y = generate(1000, num_classes, [[3.0, 0], [3.0, 3.0], [0, 3.0]], True)
        Y = Y % 2
        Y = np.reshape(Y, [-1, 1])

        _, loss_val = sess.run([train, loss], feed_dict={x: X, y: Y})
        if epoch % 100 == 0:
            print("Step:", epoch, "Current loss:", loss_val)
