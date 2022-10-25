import numpy as np


def aux_jac(x, loc):
    num = loc.shape[1]
    dif = np.tile(x[[0,2]].reshape(-1,1), (1,num)) - loc
    dif_r = np.sqrt( np.sum(dif**2., axis=0) )
    z_predic = dif_r.T;
    H = np.zeros((num, x.shape[0]))
    H[:,0] = dif[0,:] / dif_r
    H[:,2] = dif[1,:] / dif_r
    return H, z_predic


def aux_predict_meas(samp_x, loc):
    no_sns = loc.shape[1]
    no_samp = samp_x.shape[1]
    z_samp = np.zeros((no_sns, no_samp))
    for i in range(no_sns):
        dif = samp_x[[0,2],:] - np.tile(loc[:,i].reshape(-1,1), (1,no_samp))
        dif_r = np.sqrt(np.sum(dif**2., axis=0))
        z_samp[i,:] = dif_r
    return z_samp
