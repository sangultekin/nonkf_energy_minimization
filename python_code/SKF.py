import numpy as np
from AUXX import aux_jac
from AUXX import aux_predict_meas


class SKF:
    def filter(self, IN, MEAS):
        P, F, Q_CV = IN['P'], IN['F'], IN['Q_CV']
        n = IN['x_filter'].shape[1]
        
        x_bar = IN['x_filter'][:,0]
        P_bar = P
        for t in range(1,n):
            R = MEAS[t]['R']
            z_true = MEAS[t]['z_true']
            
            x_predic = F.dot(x_bar)
            P_predic = F.dot(P_bar).dot(F.T) + Q_CV
            
            P_bar = P_predic
            x_bar = x_predic
            for it in range(20):
                #reset gradients
                grd_x = np.zeros(x_bar.shape)
                grd_P = np.zeros(P_bar.shape)
                
                #use jacobian as approximation
                x_exp = x_bar
                H, z_exp = aux_jac(x_predic, MEAS[t]['loc'])
                z_g = z_true - z_exp + H.dot(x_exp)
                
                #stochastic search
                S = 500 #no samples to approximate integrals
                #note: with 20 iterations and 500 samples the total size is 1e4
                #which matches other filters' sample size
                
                P_bar = .5*(P_bar + P_bar.T)
                samp_x = np.random.multivariate_normal(x_bar, P_bar, S).T
                
                #predicted measurement
                z_samp = aux_predict_meas(samp_x, MEAS[t]['loc'])
                
                tmp1 = np.tile(z_true.reshape(-1,1), (1,S)) - z_samp
                f = np.sum(tmp1*(np.linalg.inv(R).dot(tmp1)), axis=0)
                
                tmp2 = np.tile(z_g.reshape(-1,1), (1,S)) - H.dot(samp_x)
                g = np.sum(tmp2*(np.linalg.inv(R).dot(tmp2)), axis=0)
                
                tmp3 = np.linalg.inv(P_bar).dot(samp_x) \
                    - np.tile(np.linalg.inv(P_bar).dot(x_bar).reshape(-1,1), (1,S))
                fg = np.tile((f-g).reshape(1,-1), (tmp3.shape[0],1))
                grd_x = grd_x + (1/S) * np.sum(fg*tmp3, axis=1)
                
                tmp4 = samp_x - np.tile(x_bar.reshape(-1,1), (1,S))
                fg = np.tile((f-g).reshape(1,-1), (tmp4.shape[0],1))
                tmp_mat1 = np.sign(fg)*np.sqrt(np.abs(fg)) * tmp4
                tmp_mat2 = np.sqrt(np.abs(fg)) * tmp4
                iP = np.linalg.inv(P_bar)
                T12 = np.dot(tmp_mat1, tmp_mat2.T)
                grd_P = grd_P + (1/S) * (.5*iP.dot(T12).dot(iP))
                grd_P = grd_P + (1/S) * np.sum(f-g)*(-.5*iP)
                
                #these are the gradients of the objective
                grd_P = -.5*grd_P
                grd_x = -.5*grd_x
                
                #from the gradients calculate natural gradients
                inv_Cp = P_bar
                inv_Cx = P_bar
                
                grd_x = grd_x + np.linalg.solve(P_predic, x_predic) \
                    - np.linalg.solve(P_predic, x_bar) \
                    + np.dot(np.dot(H.T, np.linalg.inv(R)),z_g) \
                    - np.dot(np.dot(np.dot(H.T,np.linalg.inv(R)),H),x_bar) 
                grd_P = grd_P + .5*np.linalg.inv(P_bar) \
                    -.5*np.linalg.inv(P_predic) \
                    -.5*np.dot(np.dot(H.T,np.linalg.inv(R)),H)
                    
                #use standard O(1/t) step size
                rho = 1/(1+it)
                
                #posterior updates with covariance check for positive definiteness
                P_prop = P_bar + rho*np.dot(np.dot(inv_Cp,grd_P),inv_Cp)
                x_prop = x_bar + rho*np.dot(inv_Cx,grd_x)
                while not np.all(np.linalg.eigvals(P_prop)>0):
                    rho = rho/2
                    P_prop = P_bar + rho*np.dot(np.dot(inv_Cp,grd_P),inv_Cp)
                    x_prop = x_bar + rho*np.dot(inv_Cx,grd_x)
                
                x_bar = x_prop
                P_bar = P_prop
                
            #state estimation
            IN['x_filter'][:,t] = x_bar
