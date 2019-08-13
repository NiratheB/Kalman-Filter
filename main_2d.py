import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INITIAL_ERR_ESTIMATE = [20, 5] # p_x, p_v
INITIAL_ESTIMATE = [4000, 280] # x, v
INITIAL_ERR_MEASUREMENT = [25, 6] #err_x, err_v
DEL_T = 1
A = np.array([[1, DEL_T],
     [0, 1]])
B = np.array([[0.5 * pow(DEL_T, 2)],[DEL_T]])
a = 2
H = np.identity(2)
d = pd.read_csv('input2.txt', sep=',', skipinitialspace=True, skip_blank_lines=True)
value_estimated = np.array([[INITIAL_ESTIMATE[0]],[INITIAL_ESTIMATE[1]]])
p = np.array([[pow(INITIAL_ERR_ESTIMATE[0], 2), 0],[0, pow(INITIAL_ERR_ESTIMATE[1],2)]])
for (index, row) in d.iterrows():
    x_measured = row['x']
    v_measured = row['v']
    value_predicted = A.dot(value_estimated) + B.dot(a)
    p = A.dot(p).dot(A.transpose())
    p = np.diag(np.diag(p))
    R = np.array([[pow(INITIAL_ERR_MEASUREMENT[0], 2), 0],[0, pow(INITIAL_ERR_MEASUREMENT[1],2)]])
    k_gain = p.dot(H) / (H.dot(p).dot(H.transpose()) + R)
    k_gain = np.diag(np.diag(k_gain))
    Y = np.array([[x_measured],[ v_measured]])
    value_estimated = value_predicted + k_gain.dot(Y- (H.dot(value_predicted)))
    p = (np.identity(2) - (k_gain.dot(H))).dot(p)
    print(f'Estimated: {value_estimated}')


