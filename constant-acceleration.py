import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def predict(x, P, F, Q):
    xpred = F.dot(x)
    Ppred = F.dot(P).dot(F.transpose()) + Q
    return (xpred, Ppred)

def innovation(xpred, Ppred, z, H, R):
    nu = z - (H.dot(xpred))
    S = R + H.dot(Ppred).dot(H.transpose())
    return (nu, S)

def innovation_update(xpred, Ppred, nu, S, H):
    K = (Ppred.dot(H.transpose()) ) / S
    xnew = xpred + (K * nu)
    Pnew = Ppred - ((K*S).dot(K.transpose()))
    return (xnew, Pnew)


delT = 1
F = np.array([[1, delT, pow(delT, 2)],
            [0, 1, delT],
            [0, 0, 1]])
B = [[0],
     [0],
     [1]]
H = np.array([[1, 0, 0]])
x = np.array([[0], [10], [20]])
P = np.array([[10, 0, 0],
              [0,10,0],
              [0,0,10]])
Q = 400* np.array([[(pow(delT, 5)/ 20),(pow(delT, 4)/8),(pow(delT, 3)/6)],
              [(pow(delT, 4)/8),(pow(delT, 3)/3),(pow(delT,2)/2)],
              [(pow(delT, 3)/6),(pow(delT, 2)/2),delT]])
R = np.array([1])
z= [2,4,6,8,10,12,14,16,18,20]
x_arr = []
k_arr = []
v_arr = []
a_arr = []

for i in range(0, len(z)):
    (xpred, Ppred) = predict(x, P, F, Q)
    (nu, S) = innovation(xpred, Ppred, z[i], H, R)
    (x, P) = innovation_update(xpred, Ppred, nu, S, H)
    x_arr.append(x[0][0])
    v_arr.append(x[1][0])
    a_arr.append(x[2][0])
    print(x)

time = range(0, len(z))
plt.plot( time, z, label='measurement')
plt.plot(time, x_arr, label= 'estimate')
plt.plot(time, v_arr, label='velocity')
plt.plot(time, a_arr, label='acceleration')
plt.legend(fontsize='x-large')
plt.show()
