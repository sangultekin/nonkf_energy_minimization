import numpy as np
from AUXX import aux_jac


class EKF:
    def filter(self, IN, MEAS):
        P, F, Q_CV = IN['P'], IN['F'], IN['Q_CV']
        n = IN['x_filter'].shape[1]
        
        for t in range(1,n):
            R = MEAS[t]['R']
            data_measurement = MEAS[t]['z_true']
            
            x_predic = F.dot(IN['x_filter'][:,t-1])
            P_predic = np.dot(np.dot(F, P), F.T) + Q_CV
            
            H, z_predic = aux_jac(x_predic, MEAS[t]['loc'])
            
            S = np.dot(np.dot(H, P_predic), H.T) + R
            K = np.dot(np.dot(P_predic, H.T), np.linalg.inv(S))
            IN['x_filter'][:,t] = x_predic + K.dot(data_measurement - z_predic)
            P = P_predic-np.dot(np.dot(K, S), K.T)
