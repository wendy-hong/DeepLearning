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


def moving_average_forecast1(series, window_size):
    """Forecasts the mean of the last few values.
       If window_size=1, then this is equivalent to naive forecast"""
    forecast = []
    for time in range(len(series) - window_size):
        forecast.append(series[time:time + window_size].mean())
    return np.array(forecast)


def moving_average_forecast2(series, window_size):
    """Forecasts the mean of the last few values.
       If window_size=1, then this is equivalent to naive forecast
       This implementation is *much* faster than the previous one"""
    mov = np.cumsum(series)
    mov[window_size:] = mov[window_size:] - mov[:-window_size]
    return mov[window_size - 1:-1] / window_size


moving_avg = moving_average_forecast2(series, 30)[split_time - 30:]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, label="Series")
plot_series(time_valid, moving_avg, label="Moving average (30 days)")
plt.show()
mae1 = tf.keras.metrics.mean_absolute_error(x_valid, moving_avg).numpy()
print(mae1)

# Using Differencing
diff_series = (series[365:] - series[:-365])
diff_time = time[365:]
diff_moving_avg = moving_average_forecast2(diff_series, 50)[split_time - 365 - 50:]
diff_moving_avg_plus_past = series[split_time - 365:-365] + diff_moving_avg
mae2 = tf.keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_past).numpy()
print(mae2)

diff_moving_avg_plus_smooth_past = moving_average_forecast2(series[split_time - 370:-359], 11) + diff_moving_avg
mae3 = tf.keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_smooth_past).numpy()
print(mae3)
