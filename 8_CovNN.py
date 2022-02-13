import cifar10.cifar10_input
import tensorflow as tf
import numpy as np

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

batch_size = 128
data_dir = 'cifar-10-batches-bin'
images_train, labels_train = cifar10.cifar10_input.inputs(eval_data=False, data_dir=data_dir, batch_size=batch_size)
images_test, labels_test = cifar10.cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)
print("Data Ready!")

# tf.nn.conv2d(input, filter, strides, paddings, use_cudnn_on_gpu=None, name=None)
# input = [batch, in_height, in_width, in_channels]
# filter = [filter_height, filter_width, in_channels, out_channels]
# tf.nn.max_pool(input, ksize, strides, padding, name=None)
# ksize = [batch=1, height, width, channels=1]  strides = [1, stride, stride, 1]
x = tf.placeholder(tf.float32, [None, 24, 24, 3])  # cifar data image of shape 24*24*3
y = tf.placeholder(tf.float32, [None, 10])  # 0-9 数字=> 10 classes
x_image = tf.reshape(x, [-1, 24, 24, 3])

W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 3, 64], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[64]))
h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 12*12

W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 64, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 6*6

W_conv3 = tf.Variable(tf.truncated_normal([5, 5, 64, 10], stddev=0.1))
b_conv3 = tf.Variable(tf.constant(0.1, shape=[10]))
h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)

nt_hpool3 = tf.nn.avg_pool2d(h_conv3, ksize=[1, 6, 6, 1], strides=[1, 6, 6, 1], padding='SAME')
nt_hpool3_flat = tf.reshape(nt_hpool3, [-1, 10])
y_conv = tf.nn.softmax(nt_hpool3_flat)

cross_entropy = -tf.reduce_sum(y * tf.math.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)
for i in range(15000):
    image_batch, label_batch = sess.run([images_train, labels_train])
    label_b = np.eye(10, dtype=float)[label_batch]  # one hot
    train_step.run(feed_dict={x: image_batch, y: label_b}, session=sess)

    if i % 200 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: image_batch, y: label_b}, session=sess)
        print("step %d, training accuracy %g" % (i, train_accuracy))

image_batch, label_batch = sess.run([images_test, labels_test])
label_b = np.eye(10, dtype=float)[label_batch]  # one hot
print("finished！ test accuracy %g" % accuracy.eval(feed_dict={x: image_batch, y: label_b}, session=sess))
