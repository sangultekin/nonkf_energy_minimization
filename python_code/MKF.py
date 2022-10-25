import numpy as np


class MKF():
    def filter(self, IN, MEAS):
        P, F, Q_CV = IN['P'], IN['F'], IN['Q_CV']
        n = IN['x_filter'].shape[1]
        
        #number of particles
        ns = 10000
        
        for t in range(1,n):
            R = MEAS[t]['R']
            data_measurement = MEAS[t]['z_true']
            dim = IN['x_filter'].shape[0]
            
            #predict step
            x_predic = F.dot(IN['x_filter'][:,t-1])
            P_predic = F.dot(P).dot(F.T) + Q_CV
            
            #use prior as proposal
            x_proposal = x_predic
            P_proposal = P_predic
            
            #sample particles
            PART = np.random.multivariate_normal(x_proposal, P_proposal, ns).T
            PART_sub = PART[[0,2],:]
            
            #predict measurements
            tmp_num = MEAS[t]['num']
            h = np.zeros((tmp_num,ns))
            for i in range(tmp_num):
                tmp_loc = MEAS[t]['loc']
                tmp_dif = PART_sub - np.tile(tmp_loc[:,i].reshape(-1,1), (1,ns))
                h[i,:] = np.sqrt(np.sum(tmp_dif**2, axis=0))
            
            #calculate importance weights
            tmp = np.tile(data_measurement.reshape(-1,1), (1,ns)) - h
            weig = -.5 * np.sum(tmp*(np.linalg.inv(R).dot(tmp)),axis=0)
            weig = weig - np.max(weig)
            weig = np.exp(weig)
            weig = weig / np.sum(weig)
            
            #collapse particles
            mu = PART.dot(weig)
            PART = PART - np.tile(mu.reshape(-1,1), (1,ns))
            Sig = np.dot((PART*np.tile(weig.reshape(1,-1), (dim,1))), PART.T)
            
            #state estimation
            IN['x_filter'][:,t] = mu
            P = Sig