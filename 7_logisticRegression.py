import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from matplotlib.colors import colorConverter, ListedColormap


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
        Y0 = np.concatenate((Y0, Y1))  # [0 0 ... 0 1 1 ... 1 2 2 ... 2 ...]

    if regression == False:  # one-hot
        class_ind = [Y0 == i for i in range(classNum)]
        # class_ind = [array([ True, ..., True, False, ..., False]), array([ True, ..., True, False, ..., False])]
        Y0 = np.stack(class_ind, 1)  # Y0 = [[True False] [True False] ... [False True]]^T  N*num_classes
        Y0 = np.asarray(Y0, dtype=np.float32)  # Y0 = [[1. 0.] [1. 0.] ... [0. 1.]]^T
    X, Y = shuffle(X0, Y0)
    return X, Y


np.random.seed(10)
input_dim = 2
num_classes = 3
lab_dim = num_classes
X, Y = generate(2000, num_classes, [[3.0, 3.0], [3.0, 0]], False)

inputs = tf.placeholder(tf.float32, [None, input_dim])
labels = tf.placeholder(tf.float32, [None, lab_dim])
W = tf.Variable(tf.random_normal([input_dim, lab_dim]), name="weight")
b = tf.Variable(tf.zeros([lab_dim]), name="bias")

output = tf.matmul(inputs, W) + b
z = tf.nn.softmax(output)
predict = tf.argmax(z, axis=1)
label = tf.argmax(labels, axis=1)

err = tf.count_nonzero(predict - label)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=output)
loss = tf.reduce_mean(cross_entropy)

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
            y1 = Y[i * minibatchSize:(i + 1) * minibatchSize, :]

            _, lossval, outputval, errval = sess.run([train, loss, output, err], feed_dict={inputs: x1, labels: y1})
            sumerr += errval
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(lossval), "err=", sumerr / minibatchSize)

    print(sess.run(W), sess.run(b))

    train_X, train_Y = generate(200, num_classes, [[3.0, 3.0], [3.0, 0]], False)
    aa = [np.argmax(l) for l in train_Y]
    colors = ['r' if l == 0 else 'b' if l == 1 else 'y' for l in aa[:]]
    plt.scatter(train_X[:, 0], train_X[:, 1], c=colors)

    nb_of_xs = 200
    xs1 = np.linspace(-1, 8, num=nb_of_xs)
    xs2 = np.linspace(-1, 8, num=nb_of_xs)
    xx, yy = np.meshgrid(xs1, xs2)  # create the grid

    classification_plane = np.zeros((nb_of_xs, nb_of_xs))
    for i in range(nb_of_xs):
        for j in range(nb_of_xs):
            classification_plane[i, j] = sess.run(predict, feed_dict={inputs: [[xx[i, j], yy[i, j]]]})

    cmap = ListedColormap([
        colorConverter.to_rgba('r', alpha=0.30),
        colorConverter.to_rgba('b', alpha=0.30),
        colorConverter.to_rgba('y', alpha=0.30)])
    plt.contourf(xx, yy, classification_plane, cmap=cmap)
    plt.show()
