import tensorflow.compat.v1 as tf
from tensorflow import keras
import numpy as np

mnist = keras.datasets.fashion_mnist
# mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))


def MultilayerPerceptron():
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(keras.layers.Dense(10, activation=tf.nn.softmax))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def ConvNN():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, (5, 5), strides=2))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(keras.layers.Dense(10, activation=tf.nn.softmax))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# model = MultilayerPerceptron()
model = ConvNN()
# model.fit(x_train, y_train, epochs=3)
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=8, batch_size=200, verbose=2)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

predictions = model.predict(x_test)
print(predictions[0])
print(np.argmax(predictions[0]), y_test[0])