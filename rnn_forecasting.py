import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)


def trend(time, slope=0):
    return slope * time


def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


def seq2vec_window_dataset(series, window_size, batch_size=32, shuffle_buffer=1000):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))  # [0 1 2 3 4]
    dataset = dataset.shuffle(shuffle_buffer)  # random
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))  # [0 1 2 3] [4]
    dataset = dataset.batch(batch_size).prefetch(1)  # x=[[5 6 7 8] [4 5 6 7]] y=[[9] [8]]
    return dataset


def seq2seq_window_dataset(series, window_size, batch_size=32, shuffle_buffer=1000):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))  # [0 1 2 3] [1 2 3 4]
    return ds.batch(batch_size).prefetch(1)  # x=[[5 6 7 8] [4 5 6 7]] y=[[6 7 8 9] [5 6 7 8]]


def sequential_window_dataset(series, window_size):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=window_size, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(window_size + 1))
    ds = ds.map(lambda window: (window[:-1], window[1:]))
    return ds.batch(1).prefetch(1)


def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast


def rnn_seq2vec(x_train, x_valid, window_size):
    train_set = seq2vec_window_dataset(x_train, window_size, batch_size=128)
    valid_set = seq2vec_window_dataset(x_valid, window_size, batch_size=128)
    model = tf.keras.models.Sequential([
        # add an extra dimension, RNN needs an input of 3 dimensions.
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
        tf.keras.layers.SimpleRNN(100, return_sequences=True),
        tf.keras.layers.SimpleRNN(100),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x * 200.0)])
    optimizer = tf.keras.optimizers.SGD(lr=1.5e-6, momentum=0.9)
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mae"])
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=50)
    # model_checkpoint = tf.keras.callbacks.ModelCheckpoint("my_checkpoint", save_best_only=True)
    model.fit(train_set, epochs=500,
              validation_data=valid_set,
              # callbacks=[early_stopping, model_checkpoint])
              callbacks=[early_stopping])
    # model = tf.keras.models.load_model("my_checkpoint")
    rnn_forecast = model_forecast(model, series[split_time - window_size:-1], window_size)[:, 0]
    mae = tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()
    return mae


def rnn_seq2seq(x_train, x_valid, window_size):
    train_set = seq2seq_window_dataset(x_train, window_size, batch_size=128)
    valid_set = seq2seq_window_dataset(x_valid, window_size, batch_size=128)
    model = tf.keras.models.Sequential([
        tf.keras.layers.SimpleRNN(100, return_sequences=True, input_shape=[None, 1]),
        tf.keras.layers.SimpleRNN(100, return_sequences=True),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x * 200.0)])
    optimizer = tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9)
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mae"])
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=10)
    model.fit(train_set, epochs=500,
              validation_data=valid_set,
              callbacks=[early_stopping])
    rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
    rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]
    mae = tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()
    return mae


class ResetStatesCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs):
        self.model.reset_states()


def rnn_stateful(x_train, x_valid, window_size):
    train_set = sequential_window_dataset(x_train, window_size)
    valid_set = sequential_window_dataset(x_valid, window_size)
    model = tf.keras.models.Sequential([
        tf.keras.layers.SimpleRNN(100, return_sequences=True, stateful=True, batch_input_shape=[1, None, 1]),
        tf.keras.layers.SimpleRNN(100, return_sequences=True, stateful=True),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x * 200.0)])
    optimizer = tf.keras.optimizers.SGD(lr=1e-7, momentum=0.9)
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mae"])
    reset_states = ResetStatesCallback()
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=50)
    model.fit(train_set, epochs=500,
              validation_data=valid_set,
              callbacks=[early_stopping, reset_states])
    model.reset_states()
    rnn_forecast = model.predict(series[np.newaxis, :, np.newaxis])
    rnn_forecast = rnn_forecast[0, split_time - 1:-1, 0]
    mae = tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()
    return mae


def rnn_lstm(x_train, x_valid, window_size):
    train_set = sequential_window_dataset(x_train, window_size)
    valid_set = sequential_window_dataset(x_valid, window_size)
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(100, return_sequences=True, stateful=True, batch_input_shape=[1, None, 1]),
        tf.keras.layers.LSTM(100, return_sequences=True, stateful=True),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x * 200.0)])
    optimizer = tf.keras.optimizers.SGD(lr=5e-7, momentum=0.9)
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mae"])
    reset_states = ResetStatesCallback()
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=50)
    model.fit(train_set, epochs=500,
              validation_data=valid_set,
              callbacks=[early_stopping, reset_states])
    rnn_forecast = model.predict(series[np.newaxis, :, np.newaxis])
    rnn_forecast = rnn_forecast[0, split_time - 1:-1, 0]
    mae = tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()
    return mae


time = np.arange(4 * 365 + 1)
slope = 0.05
baseline = 10
amplitude = 40
noise_level = 5
series = baseline \
         + trend(time, slope) \
         + seasonality(time, period=365, amplitude=amplitude) \
         + white_noise(time, noise_level, seed=42)

split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)
window_size = 30

mae_seq2vec = rnn_seq2vec(x_train, x_valid, window_size)
print('mae_seq2vec:', mae_seq2vec)
mae_seq2seq = rnn_seq2seq(x_train, x_valid, window_size)
print('mae_seq2seq:', mae_seq2seq)
mae_stateful = rnn_stateful(x_train, x_valid, window_size)
print('mae_stateful:', mae_stateful)
mae_lstm = rnn_lstm(x_train, x_valid, window_size)
print('mae_lstm:', mae_lstm)