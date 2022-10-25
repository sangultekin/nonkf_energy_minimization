#Code the for submitted paper.
#Author: San Gultekin, Yahoo! Research, san.gultekin@gmail.com
#Last date modified: 10/23/2022

import os, shutil, pickle, tqdm
import numpy as np
import pandas as pd
from EKF import EKF
from UKF import UKF
from PF import PF
from ENKF import ENKF
from SKF import SKF
from MKF import MKF
from EFKF import EFKF
import matplotlib.pyplot as plt

#define constants
T = 1
n = 300
MC_number = 100
F2 = np.array([[1., T], [0., 1.]])
F = np.r_[ np.c_[F2, np.zeros((2,2))] , np.c_[np.zeros((2,2)), F2] ]
var_CV = 1e-2
Q2 = np.array([[T**4./4, T**3./2], [T**3./2, T**2.]])
Q_CV = var_CV * np.r_[ np.c_[Q2, np.zeros((2,2))], np.c_[np.zeros((2,2)), Q2] ]
mag2 = 20**2
no_sns = 200
range_sns = 1e6
min_sns = 3
max_sns = 3

#set true to simulate parameter mismatch
mismatch = False
if mismatch:
    Q_CV = 1e-2*np.eye(4)
    
rmse_path = './rmse/'
figures_path = './figures/'
if os.path.exists('rmse'):
    shutil.rmtree('rmse')
os.mkdir('rmse')
if os.path.exists('figures'):
    shutil.rmtree('figures')
os.mkdir('figures')

#run all filters
RMSE_EKF = np.zeros(MC_number)
RMSE_UKF = np.zeros(MC_number)   
RMSE_PF = np.zeros(MC_number)
RMSE_ENKF = np.zeros(MC_number)
RMSE_SKF = np.zeros(MC_number)
RMSE_MKF = np.zeros(MC_number)
RMSE_EFKF = np.zeros(MC_number)

for m in tqdm.tqdm(range(MC_number)):
    fid = open('../data_gen/data/data_' + str(m) + '.pkl', 'rb')
    [target_state, data_full, data_actual, loc_sns, MEAS] = pickle.load(fid)
    fid.close()
    
    #prior covariance
    P = np.diag([100, .1, 100, .1])
    
    #initialize state estimates
    x_filter_EKF = np.zeros((4,n))
    x_filter_UKF = np.zeros((4,n))
    x_filter_PF = np.zeros((4,n))
    x_filter_ENKF = np.zeros((4,n))
    x_filter_SKF = np.zeros((4,n))
    x_filter_MKF = np.zeros((4,n))
    x_filter_EFKF = np.zeros((4,n))
    
    #starting point
    t = 0
    x_filter_EKF[:,t] = target_state - np.sqrt(np.diag(P)) #where is target_state?
    x_filter_UKF[:,t] = x_filter_EKF[:,t]
    x_filter_PF[:,t] = x_filter_EKF[:,t]
    x_filter_ENKF[:,t] = x_filter_EKF[:,t]
    x_filter_SKF[:,t] = x_filter_EKF[:,t]
    x_filter_MKF[:,t] = x_filter_EKF[:,t]
    x_filter_EFKF[:,t] = x_filter_EKF[:,t]

    #EKF (see the paper for abbreviations)
    IN_EKF = dict()
    IN_EKF['x_filter'] = x_filter_EKF
    IN_EKF['P'] = P
    IN_EKF['F'] = F
    IN_EKF['Q_CV'] = Q_CV
    ekf_obj = EKF()
    ekf_obj.filter(IN_EKF, MEAS)
    
    #UKF
    IN_UKF = dict()
    IN_UKF['x_filter'] = x_filter_UKF
    IN_UKF['P'] = P
    IN_UKF['F'] = F
    IN_UKF['Q_CV'] = Q_CV
    ukf_obj = UKF()
    ukf_obj.filter(IN_UKF, MEAS)
    
    #PF
    IN_PF = dict()
    IN_PF['x_filter'] = x_filter_PF
    IN_PF['P'] = P
    IN_PF['F'] = F
    IN_PF['Q_CV'] = Q_CV
    pf_obj = PF()
    pf_obj.filter(IN_PF, MEAS)
    
    #ENKF
    IN_ENKF = dict()
    IN_ENKF['x_filter'] = x_filter_ENKF
    IN_ENKF['P'] = P
    IN_ENKF['F'] = F
    IN_ENKF['Q_CV'] = Q_CV
    enkf_obj = ENKF()
    enkf_obj.filter(IN_ENKF, MEAS)
    
    #SKF
    IN_SKF = dict()
    IN_SKF['x_filter'] = x_filter_SKF
    IN_SKF['P'] = P
    IN_SKF['F'] = F
    IN_SKF['Q_CV'] = Q_CV
    skf_obj = SKF()
    skf_obj.filter(IN_SKF, MEAS)
    
    #MKF
    IN_MKF = dict()
    IN_MKF['x_filter'] = x_filter_MKF
    IN_MKF['P'] = P
    IN_MKF['F'] = F
    IN_MKF['Q_CV'] = Q_CV
    mkf_obj = MKF()
    mkf_obj.filter(IN_MKF, MEAS)
    
    #EFKF (proposed filter)
    IN_EFKF = dict()
    IN_EFKF['x_filter'] = x_filter_EFKF
    IN_EFKF['P'] = P
    IN_EFKF['F'] = F
    IN_EFKF['Q_CV'] = Q_CV
    gen_obj = EFKF()
    gen_obj.filter(IN_EFKF, MEAS, 0.75)
    
    #compute and write errors
    RMSE_EKF = np.sqrt(np.mean(np.sum((IN_EKF['x_filter'][[0,2],:]-data_actual)**2, axis=0)))
    RMSE_UKF = np.sqrt(np.mean(np.sum((IN_UKF['x_filter'][[0,2],:]-data_actual)**2, axis=0)))
    RMSE_PF = np.sqrt(np.mean(np.sum((IN_PF['x_filter'][[0,2],:]-data_actual)**2, axis=0)))
    RMSE_ENKF = np.sqrt(np.mean(np.sum((IN_ENKF['x_filter'][[0,2],:]-data_actual)**2, axis=0)))
    RMSE_SKF = np.sqrt(np.mean(np.sum((IN_SKF['x_filter'][[0,2],:]-data_actual)**2, axis=0)))
    RMSE_MKF = np.sqrt(np.mean(np.sum((IN_MKF['x_filter'][[0,2],:]-data_actual)**2, axis=0)))
    RMSE_EFKF = np.sqrt(np.mean(np.sum((IN_EFKF['x_filter'][[0,2],:]-data_actual)**2, axis=0)))
    writeLine = '{}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}'.\
        format(m, RMSE_EKF, RMSE_UKF, RMSE_PF, RMSE_ENKF, RMSE_SKF, RMSE_MKF, RMSE_EFKF)    
    with open(rmse_path + 'rmse.txt', 'a') as f:
        f.write(writeLine + '\n')

    #plot and save trajectory figures
    save_name = figures_path + 'figure' + str(m) + '.pdf'
    plt.figure()
    plt.plot(data_actual[0,:], data_actual[1,:],'black')
    plt.plot(IN_EKF['x_filter'][0,:], IN_EKF['x_filter'][2,:], 'purple')
    plt.plot(IN_PF['x_filter'][0,:], IN_PF['x_filter'][2,:], 'blue')
    plt.plot(IN_EFKF['x_filter'][0,:], IN_EFKF['x_filter'][2,:], 'red')
    plt.scatter(loc_sns[0,:], loc_sns[1,:], color='gray', s = 2)
    plt.axis('off')
    plt.legend(['Original', 'EKF', 'PF', 'EFKF'], loc='lower right')
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()

#print overall performance
column_names = ['MC_number', 'EKF', 'UKF', 'PF', 'ENKF', 'SKF', 'MKF', 'EFKF']
RMSE = pd.read_csv(rmse_path + 'rmse.txt', names = column_names)
print('\n')
print('Root Mean Square Error Results')
print('EKF RMSE: {:.4f}'.format(np.mean(np.asarray(RMSE['EKF']))))
print('UKF RMSE: {:.4f}'.format(np.mean(np.asarray(RMSE['UKF']))))
print('PF RMSE: {:.4f}'.format(np.mean(np.asarray(RMSE['PF']))))
print('NKF RMSE: {:.4f}'.format(np.mean(np.asarray(RMSE['ENKF']))))
print('SKF RMSE: {:.4f}'.format(np.mean(np.asarray(RMSE['SKF']))))
print('MKF RMSE: {:.4f}'.format(np.mean(np.asarray(RMSE['MKF']))))
print('GEN RMSE: {:.4f}'.format(np.mean(np.asarray(RMSE['EFKF']))))




