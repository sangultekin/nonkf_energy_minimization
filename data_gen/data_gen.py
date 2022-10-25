#Dataset for the submitted paper.
#Author: San Gultekin, Yahoo! Research, san.gultekin@gmail.com
#Last date modified: 10/23/2022

import os
import shutil
import pickle
import numpy as np
import matplotlib.pyplot as plt

#use this to see the trajectories generated
show_trajectories = True

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

#generate
if os.path.exists('./data'):
    shutil.rmtree('./data')
os.mkdir('./data')

for m in range(MC_number):
    outer_flag = flag_empty_sensor = False
    while not outer_flag:
        target_state = np.array([1000, 1, 1000, 1])
        data_full = np.zeros((4,n))
        data_full[:,0] = target_state
        data_actual = np.zeros((2,n))
        data_actual[:,0] = target_state[[0, 2]]
        
        #Part 1: Landscape and target.
        for i in range(1,n):
            data_full[:,i] = F.dot(data_full[:,i-1]) +\
            np.random.multivariate_normal(np.zeros(4), Q_CV)
            data_actual[0,i] = data_full[0,i]
            data_actual[1,i] = data_full[2,i]
        XMIN = np.min(data_actual[0,:]) - 50
        XMAX = np.max(data_actual[0,:]) + 50
        YMIN = np.min(data_actual[1,:]) - 50
        YMAX = np.max(data_actual[1,:]) + 50
        loc_sns_x = XMIN + (XMAX-XMIN)*np.random.rand(no_sns)
        loc_sns_y = YMIN + (YMAX-YMIN)*np.random.rand(no_sns)
        loc_sns = np.r_[loc_sns_x.reshape(1,-1), loc_sns_y.reshape(1,-1)]
        
        #Part 2: Sensor measurements.
        MEAS = [dict()]
        for i in range(1,n):
            loc_tar = data_actual[:,i]
            all_dist = loc_sns - np.tile(loc_tar.reshape(-1,1), (1,no_sns))
            all_dist = np.sqrt(np.sum(all_dist**2., axis=0))
            places = (all_dist <= range_sns)*1
            num = np.sum(places)
            if num < min_sns:
                flag_empty_sensor = True
                break
            elif num > max_sns:
                mask = np.argsort(all_dist)
                places[mask[:max_sns]] = 1
                places[mask[max_sns:]] = 0
                places = np.where(places==1)[0]
                num = len(places)
            tmp_dict = dict()
            tmp_dict['num'] = num
            tmp_dict['id'] = places
            tmp_dict['loc'] = loc_sns[:, places]
            tmp_dict['z_true'] = np.abs( all_dist[places] + np.sqrt(mag2)*np.random.randn(num) )
            tmp_dict['R'] = np.diag(mag2*np.ones(num))
            MEAS.append(tmp_dict)    
            
        #Part 3: Save.
        if not flag_empty_sensor:
            outer_flag = True
            if show_trajectories:
                plt.figure()
                plt.xlim(XMIN, XMAX)
                plt.ylim(YMIN, YMAX)
                plt.plot(data_actual[0,:], data_actual[1,:], color='k', linewidth=4)
                plt.scatter(loc_sns_x, loc_sns_y, color='purple')
                plt.title('Write Index = ' + str(m))
                plt.show()
            save_path = './data/data_' + str(m) + '.pkl'
            fid = open(save_path, 'wb')
            pickle.dump([target_state, data_full, data_actual, loc_sns, MEAS], fid, \
                        pickle.HIGHEST_PROTOCOL)
            fid.close()   
