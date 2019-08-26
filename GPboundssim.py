#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 21:43:45 2019

@author: K1898955
"""



#%% GP bounds simulation for MCAR
    
subject_vector = ['S3','S4']
sensor_vector = ['Gel','Textile']

bounds = [(1e-5,1), (1e-5,5), (1e-5,10), (1e-5, 20), (1e-5, 50), 
          (1e-5, 100), (1e-5, 1000), (1e-5, 1e5) ]


missing_rates = [0.25,0.5,0.75]

no_of_run = 20

data_dict_gp = {}


for subject in subject_vector:
    for sensor in sensor_vector:
        
        random_seed = random.random()
        
        for z in missing_rates:
            data_dict_gp[z] = {}
            for bound in bounds:
                data_dict_gp[z][bound] = final_transform(sensor, subject, trial_vector, window_size, feature_vector_list,
                                                       mi_feature_list,
                                                       missing_percentage = z, imputation_method = 'gp', 
                                                       random_state = int((z + random_seed) * 654321),
                                                       gp_ls_bounds = bound)
                
        results_dict_gp = {}
        
        for i in range(no_of_run):
            
            sim_seed = random.random()
            
            results_dict_gp[i] = pd.DataFrame(columns = bounds, index = missing_rates) 
            
            for z in missing_rates:
                
                for bound in bounds:
                    
                    results_dict_gp[i].loc[z,bound] = cv_predict_report(data_dict_gp[z][bound], 10, 0.30, param_grid, scores, 
                                                                         verbose = False, 
                                                                         random_state = int(sim_seed * 4568763))
                        
            
        save_obj(results_dict_gp, 'GP Bounds simulations results over %s runs %s %s' 
                 % (no_of_run, subject, sensor))
        



        
        
#%% GP bounds simulation for Gait
    
subject_vector = ['S3','S4']
sensor_vector = ['Gel','Textile']

bounds = [(1e-5,1), (1e-5,5), (1e-5,10), (1e-5, 20), (1e-5, 50), 
          (1e-5, 100), (1e-5, 1000), (1e-5, 1e5) ]


missing_rates = ['Gait with MI','Gait without MI']

no_of_run = 20

data_dict_gp = {}


for subject in subject_vector:
    for sensor in sensor_vector:
        
        random_seed = random.random()
        
        for z in missing_rates:
            data_dict_gp[z] = {}
            for bound in bounds:
                data_dict_gp[z][bound] = final_transform(sensor, subject, trial_vector, window_size, feature_vector_list,
                                                       mi_feature_list,
                                                       missing_percentage = z, imputation_method = 'gp', 
                                                       random_state = None,
                                                       gp_ls_bounds = bound)
                
        results_dict_gp = {}
        
        for i in range(no_of_run):
            
            sim_seed = random.random()
            
            results_dict_gp[i] = pd.DataFrame(columns = bounds, index = missing_rates) 
            
            for z in missing_rates:
                
                for bound in bounds:
                    
                    results_dict_gp[i].loc[z,bound] = cv_predict_report(data_dict_gp[z][bound], 10, 0.30, param_grid, scores, 
                                                                         verbose = False, 
                                                                         random_state = int(sim_seed * 4568763))
                        
            
        save_obj(results_dict_gp, 'Gait GP Bounds simulations results over %s runs %s %s' 
                 % (no_of_run, subject, sensor))


    
rmtree(cachedir)


# After simulations, use the function get_gp_bounds to get the optimal bounds.
