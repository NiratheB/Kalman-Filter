import matplotlib.pyplot as plt
import numpy as np

INITIAL_ERR_ESTIMATE = 68
INITIAL_ESTIMATE = 5
INITIAL_ERR_MEASUREMENT = 4


def calculate_kalman_gain(err_estimate, err_data):
    return (err_estimate /(err_estimate+ err_data))


def get_estimate(k_gain, estimate, data_measured):
    return estimate + (k_gain *(data_measured -estimate))


def get_err_estimate(err_estimate,k_gain):
    return ((1-k_gain) * err_estimate)


measurement_array = []
estimate_array = []
with open('input.txt', 'r') as f:
    err_estimate = INITIAL_ERR_ESTIMATE
    estimate = INITIAL_ESTIMATE
    err_measurement = INITIAL_ERR_MEASUREMENT
    for line in f:
        data_measured = int(line)
        measurement_array.append(data_measured)
        k_gain = calculate_kalman_gain(err_estimate, err_measurement)
        estimate = get_estimate(k_gain, estimate, data_measured)
        estimate_array.append(estimate)
        err_estimate = get_err_estimate(err_estimate, k_gain)
        print(f'Data Measured : {data_measured} Estimate: {estimate} K_GAIN: {k_gain} Error: {err_estimate}')


print("End")
time = range(0, len(measurement_array))
plt.plot( time, measurement_array, label='measurement')
plt.plot(time, estimate_array, label= 'estimate')
plt.show()

