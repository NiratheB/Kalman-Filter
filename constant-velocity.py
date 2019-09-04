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
Q = np.array([[10,0], [0, 10]])
R = np.array([1])
#z = [2.5, 1 ,4 ,2.5 ,5.5] # positions measured
z= [2.08,4.1,3,6.2,8.8,10.09,12.3,14.8,16.87,18.1,20.99]
x_arr = []
v_arr = []
k_arr = []

for i in range(0, len(z)):
    (xpred, Ppred) = predict(x, P, F, Q)
    (nu, S) = innovation(xpred, Ppred, z[i], H, R)
    (x, P) = innovation_update(xpred, Ppred, nu, S, H)
    x_arr.append(x[0][0])
    v_arr.append(x[1][0])
    print(x)

time = range(0, len(z))
plt.plot( time, z, label='measurement')
plt.plot(time, x_arr, label= 'estimate')
plt.plot(time, v_arr, label='velocity')
plt.legend(fontsize='x-large')
plt.show()
