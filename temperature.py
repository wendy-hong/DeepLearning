import tensorflow.compat.v1 as tf
import numpy as np

# import matplotlib.pyplot as plt

# 1. Set up training data
celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# 2. Create the model and Assemble layers into the model
# tf.keras.layers.Dense(units,activation=None,use_bias=True,
#                    kernel_initializer="glorot_uniform",bias_initializer="zeros",
#                    kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,
#                    kernel_constraint=None,bias_constraint=None,**kwargs)
# output = activation(dot(input, kernel) + bias)
l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
# tf.keras.Sequential(layers=None, name=None)
model = tf.keras.Sequential([l0])
# model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])

# 3. Compile the model, with loss and optimizer functions
# Model.compile(optimizer="rmsprop",loss=None,
#               metrics=None,loss_weights=None,weighted_metrics=None,
#               run_eagerly=None,steps_per_execution=None,**kwargs)
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.1))

# 4. Train the model
# Model.fit(x=None,y=None,batch_size=None,epochs=1,verbose="auto",
#           callbacks=None,validation_split=0.0,validation_data=None,shuffle=True,
#           class_weight=None,sample_weight=None,initial_epoch=0,steps_per_epoch=None,
#           validation_steps=None,validation_batch_size=None,validation_freq=1,
#           max_queue_size=10,workers=1,use_multiprocessing=False)
# One epoch is a full iteration of the data examples.
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=0)
print("Finished training the model")

# plt.xlabel('Epoch Number')
# plt.ylabel("Loss Magnitude")
# plt.plot(history.history['loss'])
# plt.show()

# 5. Predict
# Model.predict(x,batch_size=None,verbose=0,steps=None,callbacks=None,
#               max_queue_size=10,workers=1,use_multiprocessing=False,)
print(model.predict([100.0]))

print("layer variables (weights) : {}".format(l0.get_weights()))
