import numpy as np


class ENKF:
    def filter(self, IN, MEAS):
        P, F, Q_CV = IN['P'], IN['F'], IN['Q_CV']
        n = IN['x_filter'].shape[1]
        
        ns = 10000
        PART = np.random.multivariate_normal(IN['x_filter'][:,0], P, ns).T
        
        for t in range(1,n):
            R = MEAS[t]['R']
            data_measurement = MEAS[t]['z_true']
            dim = IN['x_filter'].shape[0]
            
            PART = F.dot(PART) + np.random.multivariate_normal(np.zeros(dim), Q_CV, ns).T
            PART_sub = PART[[0, 2], :]
            
            tmp_num = MEAS[t]['num']
            h = np.zeros((tmp_num, ns))
            for i in range(tmp_num):
                tmp_loc = MEAS[t]['loc']
                tmp_dif = PART_sub - np.tile(tmp_loc[:,i].reshape(-1,1), (1,ns))
                h[i,:] = np.sqrt(np.sum(tmp_dif**2., axis=0))
                
            #data assimilation
            PART_Y = h + np.sqrt(R).dot(np.random.randn(h.shape[0], h.shape[1]))
            REP_Y = np.tile(data_measurement.reshape(-1,1), (1,ns))
            
            TMP = np.sum(PART, axis=1)
            TMP = np.tile(TMP.reshape(-1,1), (1,PART.shape[1])) / ns
            EXf = PART - TMP
            TMP = np.sum(PART_Y, axis=1)
            TMP = np.tile(TMP.reshape(-1,1), (1,PART_Y.shape[1])) / ns
            EYf = PART_Y - TMP
            
            PXX = np.dot(EXf, EXf.T) / (ns-1)
            PYY = np.dot(EYf, EYf.T) / (ns-1)
            PXY = np.dot(EXf, EYf.T) / (ns-1)
            
            K = PXY.dot(np.linalg.inv(PYY))
            
            PART = PART + K.dot(REP_Y - PART_Y)
            
            IN['x_filter'][:,t] = np.sum(PART, axis=1) / ns
            
            
            
            
            