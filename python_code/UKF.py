import numpy as np


class UKF:
    def filter(self, IN, MEAS):
        P, F, Q_CV = IN['P'], IN['F'], IN['Q_CV']
        n = IN['x_filter'].shape[1]
                
        #UT parameters
        npt = 4
        lamda = 2-npt
        Wm = 1/(2*(npt+lamda)) * np.ones(2*npt+1)
        Wc = 1/(2*(npt+lamda)) * np.ones(2*npt+1)
        Wm[0] = lamda/(npt+lamda)
        Wc[0] = lamda/(npt+lamda)
        
        for t in range(1,n):
            R = MEAS[t]['R']
            data_measurement = MEAS[t]['z_true']
            dim = IN['x_filter'].shape[0];
            dim2 = MEAS[t]['num'];
            
            #generate sigma points
            P = .5*(P + P.T)
            cho1 = np.linalg.cholesky(P)
            x_expand = np.tile(IN['x_filter'][:,t-1].reshape(-1,1), (1,npt))
            xgamaP1 = x_expand + np.sqrt(npt+lamda)*cho1
            xgamaP2 = x_expand - np.sqrt(npt+lamda)*cho1
            Xsigma = np.c_[IN['x_filter'][:,[t-1]], xgamaP1, xgamaP2]
            
            #predict step
            Xsigma_predic= F.dot(Xsigma);
            Wm_expand = np.tile(Wm.reshape(1,-1), (Xsigma_predic.shape[0],1))
            x_predic = np.sum(Wm_expand*Xsigma_predic, axis=1)
            P_predic = np.zeros(P.shape)
            for j in range(2*npt+1):
                P_predic = P_predic + Wc[j]*\
                np.outer(Xsigma_predic[:,j]-x_predic, Xsigma_predic[:,j]-x_predic)
            P_predic = P_predic + Q_CV
            
            #predict measurements from sigma points
            Zsigma_predic = np.zeros((dim2,2*npt+1))
            for j in range(2*npt+1):
                tmp_vec = Xsigma_predic[[0, 2], j];
                tmp_dif = np.tile(tmp_vec.reshape(-1,1),(1,dim2)) - MEAS[t]['loc']
                Zsigma_predic[:,j] = np.sqrt(np.sum(tmp_dif**2.,axis=0))
            Wm_expand = np.tile(Wm.reshape(1,-1), (Zsigma_predic.shape[0],1))
            z_predic = np.sum(Wm_expand*Zsigma_predic, axis=1)

            #update step
            Pzz = np.zeros((dim2,dim2))
            for j in range(2*npt+1):
                Pzz = Pzz + Wc[j]*\
                np.outer((Zsigma_predic[:,j]-z_predic), (Zsigma_predic[:,j]-z_predic).T)
            Pzz = Pzz + R
            
            Pxz = np.zeros((dim,dim2));
            for j in range(2*npt+1):
                Pxz = Pxz + Wc[j]*\
                np.outer((Xsigma_predic[:,j]-x_predic), (Zsigma_predic[:,j]-z_predic))
            
            K = np.dot(Pxz, np.linalg.inv(Pzz));
            IN['x_filter'][:,t] = x_predic + K.dot(data_measurement-z_predic);
            P = P_predic - np.dot(np.dot(K, Pzz), K.T)
            
            
            
            
            