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


def seq2seq_window_dataset(series, window_size, batch_size=32, shuffle_buffer=1000):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)


def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast


def cnn_lstm(x_train, x_valid, window_size):
    train_set = seq2seq_window_dataset(x_train, window_size, batch_size=128)
    valid_set = seq2seq_window_dataset(x_valid, window_size, batch_size=128)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding="causal",
                               activation="relu", input_shape=[None, 1]),
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x * 200)])
    optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mae"])
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=10)
    model.fit(train_set, epochs=500,
              validation_data=valid_set,
              callbacks=[early_stopping])
    rnn_forecast = model_forecast(model, series[:, np.newaxis], window_size)
    rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]
    mae = tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()
    return mae


def cnn_wavenet(x_train, x_valid, window_size):
    train_set = seq2seq_window_dataset(x_train, window_size, batch_size=128)
    valid_set = seq2seq_window_dataset(x_valid, window_size, batch_size=128)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=[None, 1]))
    for dilation_rate in (1, 2, 4, 8, 16, 32):
        model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=2, strides=1,
                                         dilation_rate=dilation_rate, padding="causal", activation="relu"))
    model.add(tf.keras.layers.Conv1D(filters=1, kernel_size=1))
    optimizer = tf.keras.optimizers.Adam(lr=3e-4)
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mae"])
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=10)
    model.fit(train_set, epochs=500,
              validation_data=valid_set,
              callbacks=[early_stopping])
    cnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
    cnn_forecast = cnn_forecast[split_time - window_size:-1, -1, 0]
    mae = tf.keras.metrics.mean_absolute_error(x_valid, cnn_forecast).numpy()
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

mae_cnn_lstm = cnn_lstm(x_train, x_valid, window_size)
print('mae_cnn_lstm:', mae_cnn_lstm)
mae_cnn_wavenet = cnn_wavenet(x_train, x_valid, window_size)
print('mae_cnn_wavenet:', mae_cnn_wavenet)
