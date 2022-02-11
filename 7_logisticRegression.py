import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle


def generate(sample_size, mean, cov, diff, regression):
    classNum = 2
    samples_per_class = sample_size // 2
    X0 = np.random.multivariate_normal(mean, cov, samples_per_class)
    Y0 = np.zeros(samples_per_class)

    for ci, d in enumerate(diff):
        X1 = np.random.multivariate_normal(mean + d, cov, samples_per_class)
        Y1 = (ci + 1) * np.ones(samples_per_class)
        X0 = np.concatenate((X0, X1))
        Y0 = np.concatenate((Y0, Y1))  # [0 0 ... 0 1 1 ... 1 2 2 ... 2 ...]

    if regression == False:
        print("ssss")
        class_ind = [Y0 == i for i in range(
            classNum)]  # [array([ True, ..., True, False, ..., False]), array([ True, ..., True, False, ..., False])]
        Y0 = np.asarray(np.stack(class_ind, 1), dtype=np.float32)
    X, Y = shuffle(X0, Y0)  # random arrangement
    return X, Y


input_dim = 2
lab_dim = 1
np.random.seed(10)
num_classes = 2
mean = np.random.randn(num_classes)
cov = np.eye(num_classes)
X, Y = generate(1000, mean, cov, [3.0], True)

inputs = tf.placeholder(tf.float32, [None, input_dim])
labels = tf.placeholder(tf.float32, [None, lab_dim])
W = tf.Variable(tf.random_normal([input_dim, lab_dim]), name="weight")
b = tf.Variable(tf.zeros([lab_dim]), name="bias")

output = tf.nn.sigmoid(tf.matmul(inputs, W) + b)
cross_entropy = -(labels * tf.log(output) + (1 - labels) * tf.log(1 - output))
loss = tf.reduce_mean(cross_entropy)
sqerr = tf.square(labels - output)
mse = tf.reduce_mean(sqerr)
optimizer = tf.train.AdamOptimizer(0.04)
train = optimizer.minimize(loss)

Epoches = 50
minibatchSize = 25

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(Epoches):
        sumerr = 0
        for i in range(len(Y) // minibatchSize):
            x1 = X[i * minibatchSize:(i + 1) * minibatchSize, :]
            y1 = np.reshape(Y[i * minibatchSize:(i + 1) * minibatchSize], [-1, 1])
            tf.reshape(y1, [-1, 1])
            _, lossval, outputval, errval = sess.run([train, loss, output, mse], feed_dict={inputs: x1, labels: y1})
            sumerr += errval
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(lossval), "err=", sumerr / minibatchSize)
