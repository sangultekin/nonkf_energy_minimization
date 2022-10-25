import autograd.numpy as np
from autograd.numpy.linalg import det, inv
from autograd import grad
#from scipy.special import logsumexp


class EFKF:    
    def enlb(self, mu, Sig, mu0, Sig0, B, y, MEAS, alpha):
        dim = mu.shape[0]
        
        Sig_chol = np.linalg.cholesky(Sig)
        MU = np.tile(mu.reshape(-1,1), (1,B.shape[1]))
        X = np.dot(Sig_chol, B) + MU
        C = np.array([[1,0,0,0],[0,0,1,0]])
        CX = np.dot(C, X)
        LOC0 = np.tile(MEAS['loc'][:,0].reshape(-1,1), (1, B.shape[1]))
        LOC1 = np.tile(MEAS['loc'][:,1].reshape(-1,1), (1, B.shape[1]))
        LOC2 = np.tile(MEAS['loc'][:,2].reshape(-1,1), (1, B.shape[1]))
        h1 = np.sqrt(np.sum((CX-LOC0)**2,axis=0,keepdims=True))
        h2 = np.sqrt(np.sum((CX-LOC1)**2,axis=0,keepdims=True))
        h3 = np.sqrt(np.sum((CX-LOC2)**2,axis=0,keepdims=True))
        H = np.concatenate([h1, h2, h3])
                
        Y = np.tile(y.reshape(-1,1), (1,B.shape[1]))
        
        iR = inv(MEAS['R'])
        T1 = -.5*dim*np.log(2*np.pi) -.5*np.log(1e-100 + det(MEAS['R']))
        T1 = T1 - .5*np.sum((Y-H)*np.dot(iR,(Y-H)), axis=0)
        
        iSig0 = inv(Sig0)
        MU0 = np.tile(mu0.reshape(-1,1), (1,B.shape[1]))
        T2 = -.5*np.sum(X*np.dot(iSig0,X), axis=0) + np.sum(X*np.dot(iSig0,MU0), axis=0)
        
        iSig = inv(Sig)
        MU = np.tile(mu.reshape(-1,1), (1,B.shape[1]))
        T3 = -.5*np.sum(X*np.dot(iSig,X), axis=0) + np.sum(X*np.dot(iSig,MU), axis=0)
        
        T = alpha*(T1+T2-T3)
        T_max = np.max(T)
        TR = np.exp(T-T_max)
                
        ans = -.5*np.dot(mu,np.dot(iSig, mu)) -.5*np.log(1e-100 + det(Sig)) 
        ans = ans - (1./alpha)*np.log(1e-100 + np.sum(TR)/B.shape[1]) - (1./alpha)*T_max
        return ans
        

    def filter(self, IN, MEAS, alpha):
        P, F, Q_CV = IN['P'], IN['F'], IN['Q_CV']
        n = IN['x_filter'].shape[1]
        
        x_bar = IN['x_filter'][:,0]
        P_bar = P
        for t in range(1,n):
            R = MEAS[t]['R']
            z_true = MEAS[t]['z_true']
            y = z_true
            
            x_predic = F.dot(x_bar)
            P_predic = F.dot(P_bar).dot(F.T) + Q_CV
            
            P_bar = P_predic
            x_bar = x_predic
            for it in range(20):                
                S = 500
                B = np.random.randn(x_bar.shape[0], S)
                
                #make functions
                def vlb_x(x):
                    #return self.vlb(x, P_bar, x_predic, P_predic, B, y, MEAS[t]) 
                    return self.enlb(x, P_bar, x_predic, P_predic, B, y, MEAS[t], alpha)   
                def vlb_P(P):
                    #return self.vlb(x_bar, P, x_predic, P_predic, B, y, MEAS[t])
                    return self.enlb(x_bar, P, x_predic, P_predic, B, y, MEAS[t], alpha)
                
                #make gradients
                grdfun_x = grad(vlb_x)
                grdfun_P = grad(vlb_P)
                grd_x = grdfun_x(x_bar)
                grd_P = grdfun_P(P_bar)
                grd_P = .5 * (grd_P + grd_P.T)
                
                #from the gradients calculate natural gradients
                #this is necessary for covariance stability
                inv_Cp = P_bar
                inv_Cx = P_bar
                
                #use standard O(1/t) step size
                rho = 1/(1+it)
                
                #posterior updates with covariance check for positive definiteness
                P_prop = P_bar - rho*np.dot(np.dot(inv_Cp,grd_P),inv_Cp)
                x_prop = x_bar - rho*np.dot(inv_Cx,grd_x)
                while not np.all(np.linalg.eigvals(P_prop)>0):
                    rho = rho/2
                    P_prop = P_bar - rho*np.dot(np.dot(inv_Cp,grd_P),inv_Cp)
                    x_prop = x_bar - rho*np.dot(inv_Cx,grd_x)
                
                x_bar = x_prop
                P_bar = P_prop
                P_bar = .5 * (P_bar + P_bar.T)
            
            #state estimation
            IN['x_filter'][:,t] = x_bar
            
            
            
            
            