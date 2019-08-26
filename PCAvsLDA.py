#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 21:43:06 2019

@author: K1898955
"""

#%% Simulations PCA vs LDA


dim_method = ['LDA','PCA']

acc_over_method = {}
std_over_method = {}

no_of_runs = 50

for method in dim_method:
    
    accuracy_scores = []
    
    features = final_transform('Gel', 'S4', trial_vector, window_size, feature_vector_list,
                               mi_feature_list, missing_percentage = 0 , imputation_method = 'knn',
                               gp_ls_bounds = (1e-5,1e5),
                               random_state = 0)
    
    for a in range(0, no_of_runs):
        
        sim_seed = random.random()
        accuracy_scores.append(cv_predict_report(features, 10, 0.30, param_grid, scores, dim_reduction = method,
                                                     verbose = False, random_state = int(sim_seed * 123123123)))
        
    acc_over_method[method] = np.array(accuracy_scores)




# t-test

ttest_ind(acc_over_method['LDA'], acc_over_method['PCA'], equal_var = False)

print(acc_over_method['LDA'].std() * 100)
print(acc_over_method['LDA'].mean() * 100)

print(acc_over_method['PCA'].std() * 100)
print(acc_over_method['PCA'].mean() * 100)