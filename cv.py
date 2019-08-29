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
F = np.array([[1, delT],
     [0, 1]])
H = np.array([[1, 0]])
x = np.array([[0], [10]])
P = np.array([[10, 0],
     [0,10]])
Q = np.array([[1,0], [0, 1]])
R = np.array([1])
z = [2.5, 1 ,4 ,2.5 ,5.5]
x_arr = []
k_arr = []

for i in range(0, 5):
    (xpred, Ppred) = predict(x, P, F, Q)
    (nu, S) = innovation(xpred, Ppred, z[i], H, R)
    (x, P) = innovation_update(xpred, Ppred, nu, S, H)
    x_arr.append(x[0][0])

time = range(0, 5)
plt.plot( time, z, label='measurement')
plt.plot(time, x_arr, label= 'estimate')
plt.show()
