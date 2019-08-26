#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 21:59:09 2019

@author: K1898955
"""





#%% Gait assumptions simulations:

#Note: K-NN imputation does not work for variables with higher than 80% missing rate.
# Possible causes: data is then too sparse to find enough neighbors.

missing_rates = [0,'Gait without MI', 'Gait with MI',]
acc_over_missing_rates = {}
std_over_missing_rates = {}


imputation_methods = ['zero','mean','knn','gp']
#imputation_methods = ['zero']

subject_vector = ['S4', 'S3']
sensor_vector = ['Gel', 'Textile']
no_of_run = 20


gp_bounds = (1e-5, 1e5)

for subject in subject_vector:
    for sensor in sensor_vector:
        
        data_dict = {}        
        
        # Imputation steps:
        for z in missing_rates:
            
            if z != 0 and imp == 'gp':
                gp_bounds = get_gp_bounds(subject, sensor, z)
            
            data_dict[z] = {}
            
            for imp in imputation_methods:
                
                
                data_dict[z][imp] = final_transform(sensor, subject, trial_vector, window_size, 
                                                   feature_vector_list ,
                                                   mi_feature_list, missing_percentage = z, 
                                                   imputation_method = imp,gp_ls_bounds = gp_bounds)
        
        
        
        # Classification results step:
        
        results_dict = {}
        
        for i in range(no_of_run):
            
            results_dict[i] = pd.DataFrame(columns = imputation_methods, index = missing_rates)
            
            sim_seed = random.random()
            
            for z in missing_rates:
                
                for imp in imputation_methods:
            
                    results_dict[i].loc[z,imp] = cv_predict_report(data_dict[z][imp], 10, 0.30, param_grid, scores, 
                                                                        verbose = False, dim_reduction = 'PCA',
                                                                        random_state = int(sim_seed * 100000))
                    
        
        
        rmtree(cachedir)


        # Save results
        
        save_obj(results_dict, 'Gait simulations results %s %s %s'
                 % (subject,sensor, imputation_methods))




#%% Plotting:

subject_vector = ['S3','S4']
sensor_vector = ['Gel','Textile']

imputation_methods = ['zero','mean','knn','gp']
missing_rates = [0, 'Gait without MI', 'Gait with MI',]

results_dict = load_obj('Gait simulations results %s %s %s'
                        % (subject,sensor, imputation_methods))


for subject in subject_vector:
    
    for sensor in sensor_vector:
        
        results_dict = load_obj('Gait simulations results %s %s %s'
                                % (subject,sensor, imputation_methods))

        

        mean_acc = avg_dict_df(results_dict)
        
        mean_acc = mean_acc.round(4) * 100
        

        imputation_methods = ['zero','mean','knn','gp']
        markers = ['o','s','D','P']
        
        plt.figure()
        
        for imp,marker in zip(imputation_methods,markers):
            plt.errorbar([0,1,2], mean_acc[imp],
                         label =  str(imp).upper() + ' imputation' if imp == 'gp' or imp == 'knn' or imp == 'pmm' else str(imp).capitalize() + ' imputation', 
                         capsize = 5, capthick = 3, marker = marker,
                         markersize = 12, linewidth = 2)
        
        plt.legend()
        plt.ylabel('Mean balanced accuracy (%)')
        plt.xticks([0,1,2],missing_rates)
        plt.grid(axis = 'y', linestyle = '--')
        
        
        plt.savefig('Results Figures/%s%sClassificationGait.pdf' % (subject, sensor), dpi=300)


#%% Output latex tables:


no_of_run = 20
missing_rates = [0, 'Gait without MI', 'Gait with MI']


subject_vector = ['S3','S4']
sensor_vector = ['Gel','Textile']

for subject in subject_vector:
    for sensor in sensor_vector:
        
        imputation_methods = ['zero','mean','knn','gp']
        
        acc_1 = load_obj('Gait simulations results %s %s %s'
                         % (subject,sensor, imputation_methods))

        
        mean_acc = avg_dict_df(acc_1)
        mean_acc = mean_acc.round(2)
        
        
        mean_acc.columns = ['Zero', 'Mean','K-NN', 'STGP']
        
        # Get standard deviation
        
        std = std_dict_df(acc_1, imputation_methods, missing_rates = missing_rates)

        std.columns = ['Zero', 'Mean','K-NN', 'STGP']
        
        combine = combine_acc_std(mean_acc, std, missing_rates )
        
        print(combine)
        
        with open('tables/%s%sGait.tex' % (subject,sensor), 'w') as tb:
            tb.write(combine.to_latex(escape = False))
            
