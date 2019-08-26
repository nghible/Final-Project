#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 20:25:46 2019

@author: K1898955
"""
#%%



with open('S4Gel_t1.txt') as file:
    data = pd.read_csv(file)
    

imputation_methods = ['zero','mean','knn','gp','pmm']

missing_rates = [0.25, 0.5, 0.75]


    
comp_results = pd.DataFrame(columns = imputation_methods, index = missing_rates)
#%%

comp_results
#%%


for z in missing_rates:
    for imp in imputation_methods:
        
        data = take_out_random(data, z, random_state = int(z * 65748))
        
        t0 = time()
        data = impute(data, imp, gp_ls_bounds = get_gp_bounds('S4', 'Gel', z ))
        t1 = time()
        
        comp_results.loc[z, imp] = int(t1 - t0)
        

        
#%%
comp_results = abs(comp_results)


with open('tables/CompTable.tex', 'w') as tb:
    tb.write(comp_results.to_latex(escape = False))