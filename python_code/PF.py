import numpy as np


class PF:
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

            tmp = np.tile(data_measurement.reshape(-1,1), (1,ns)) - h
            weig = -.5 * np.sum(tmp * np.linalg.inv(R).dot(tmp), axis=0)
            weig = weig - np.max(weig)
            weig = np.exp(weig)
            weig = weig / np.sum(weig)

            weig_expand = np.tile(weig.reshape(1,-1),(dim,1))
            IN['x_filter'][:,t] = np.sum(weig_expand*PART, axis=1)

            flag_resample = True
            if flag_resample:
                PART_new = np.zeros(PART.shape)
                weig_new = np.zeros(ns)
                weig_cdf = np.cumsum(weig)
                
                i = 0
                u = np.zeros(ns)
                u[0] = np.random.rand(1)/ns
                for j in range(ns):
                    u[j] = u[0] + (j-1)/ns
                    while u[j] > weig_cdf[i]:
                        i+=1
                    PART_new[:,j] = PART[:,i]
                    weig_new[j] = 1/ns
                PART = PART_new
                weig = weig_new
