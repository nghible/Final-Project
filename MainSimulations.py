#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 17:23:00 2019

@author: K1898955
"""


#%% MCAR Simulation


#Note: K-NN imputation does not work for variables with higher than 80% missing rate.
# Possible causes: data is then too sparse to find enough neighbors.

missing_rates = [0, 0.25, 0.5, 0.75]
#missing_rates = [0]
acc_over_missing_rates = {}
std_over_missing_rates = {}

# 100 is a bit excessive. This can takes about 1 hour to run.
no_of_run = 20

imputation_methods = ['zero','mean','knn','gp','pmm']

#default value
gp_bounds = (1e-5, 5)



subject_vector = ['S3', 'S4']
sensor_vector = ['Textile', 'Gel']

# This experiment ensure that for each random_state in final transform, every
# imputation methods will deal with the same missingness patterns. If we want to
# simulate for different missingnes patterns, but still keep the fact that every methods
# have to impute on the same patterns, then randomize the sim_seed, 
# but do NOT put None into random_state. 

for subject in subject_vector:
    for sensor in sensor_vector:
        
        df_acc_over_rates = {}
        
        for i in range(no_of_run):
            
            sim_seed = random.random()
            
            for imp in imputation_methods:
                
                acc_over_missing_rates[imp] = []
                
                for z in missing_rates:
                    
                    if z != 0:
                        gp_bounds = get_gp_bounds(subject,sensor,z)
                    
                    # Random-state is controlling the random missing data.
                    data = final_transform(sensor, subject, trial_vector, window_size, feature_vector_list,
                                           mi_feature_list, missing_percentage = z, 
                                           imputation_method = imp, gp_ls_bounds = gp_bounds,
                                           random_state = int((z + sim_seed) * 1000))
                    
                    acc_over_missing_rates[imp].append(cv_predict_report(data, 10, 0.30, param_grid, scores, 
                                                        verbose = False, dim_reduction = 'PCA',
                                                        random_state = int(sim_seed * 100000)))
                    
            df_acc_over_rates[i] = pd.DataFrame.from_dict(acc_over_missing_rates)
            
            print()
            print()
            print('####################Run: %s completed###################' % (i + 1))
            print()
            print()
        
        rmtree(cachedir)
        
        
        # Save objects
        
        save_obj(df_acc_over_rates, 'MCAR results %s over %s runs %s %s' 
                 % (imputation_methods, no_of_run, subject, sensor))
        


        
#%% Plotting:

no_of_run = 20
missing_rates = [0, 0.25, 0.5, 0.75]


subject_vector = ['S4','S3']
sensor_vector = ['Gel', 'Textile']


            
for subject in subject_vector:
    
    for sensor in sensor_vector:
        
        no_of_run = 20
        
        imputation_methods = ['zero','mean','knn','gp','pmm']
        
        acc = load_obj('MCAR results %s over %s runs %s %s' 
                         % (imputation_methods, no_of_run, subject, sensor))
        
        mean_acc = avg_dict_df(acc)
        
        
        mean_acc = mean_acc.round(4) * 100
        

        imputation_methods = ['zero','mean','knn','gp','pmm']
        markers = ['o','s','D','P', '^']
        
        plt.figure()
        
        for imp,marker in zip(imputation_methods,markers):
            plt.errorbar(np.array(missing_rates) * 100, mean_acc[imp],
                         label =  str(imp).upper() + ' imputation' if imp == 'gp' or imp == 'knn' or imp == 'pmm' else str(imp).capitalize() + ' imputation', 
                         capsize = 5, capthick = 3, marker = marker,
                         markersize = 12, linewidth = 2)
        
        plt.legend()
        plt.xlabel('Missing Rates (%)')
        plt.ylabel('Mean balanced accuracy (%)')
        plt.xticks(np.array(missing_rates) * 100)
        plt.grid(axis = 'y', linestyle = '--')
        
        plt.savefig('Results Figures/%s%sClassificationMCAR.pdf' % (subject, sensor), dpi=300)





#%% Output latex tables:
        
no_of_run = 20
missing_rates = [0, 0.25, 0.5, 0.75]


subject_vector = ['S3','S4']
sensor_vector = ['Gel','Textile']

for subject in subject_vector:
    for sensor in sensor_vector:
        
        imputation_methods = ['zero','mean','knn','gp','pmm']
        
        acc = load_obj('MCAR results %s over %s runs %s %s' 
                       % (imputation_methods, no_of_run, subject, sensor))
        
        
        
        
        mean_acc = avg_dict_df(acc)


        
        mean_acc.columns = ['Zero', 'Mean','K-NN', 'STGP', 'PMM']
        
        # Get standard deviation
        
        std = std_dict_df(acc,imputation_methods)

        
        std.columns = ['Zero', 'Mean','K-NN', 'STGP','PMM']
        
        combine = combine_acc_std(mean_acc, std ,missing_rates = missing_rates)
        
        print(combine)
        
        with open('tables/%s%sMCAR.tex' % (subject,sensor), 'w') as tb:
            tb.write(combine.to_latex(escape = False))
            




