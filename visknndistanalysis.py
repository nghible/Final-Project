#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 21:42:00 2019

@author: K1898955
"""


#%% Example plots of missing data under gait assump.
    

with open('S3Textile_t1.txt') as file:
    data = pd.read_csv(file, header = 0)
    
data = preprocess(data)

data = take_out_data(data)


# Plot missing data.


for c,col in zip([0,1,2,3],color):
    plt.figure()
    data.iloc[:,c].plot()
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (Volts)')
    plt.grid(axis = 'x', linestyle = '--')
    plt.savefig('Results Figures/S3TextileChannel%sGait.pdf' % (c + 1), dpi=300)



#%% Pair plots with regression line:
    

with open('S4Gel_t1.txt') as file:
    data = pd.read_csv(file, header = 0)
    
data = preprocess(data)

data.columns = ['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4', 'Class']
grid = sns.pairplot(data, kind = 'reg', markers = '+',
                    vars= ['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4'])


grid.savefig('Results Figures/PairPlotS4Geltrial1.pdf', dpi = 300)


#%% Distributions comparison plot before and after imputation:



with open('S3Textile_t1.txt') as file:
    data = pd.read_csv(file, header = 0)
    
data = preprocess(data)

mi_data = data.copy()

mi_data = take_out_random(mi_data, 0.5)

mi_data = impute(mi_data, 'knn')

sns.distplot(data['data_1'], label = 'Full data',
             axlabel = False)
sns.distplot(mi_data['data_1'], color = 'tab:red', 
             label = '3-NN imputed MCAR at 50% data',
             axlabel = False)

plt.xlim([-0.06, 0.06])
sns.despine()
plt.legend(loc='best', fontsize = 'small',frameon = False)
plt.savefig('Results Figures/DistsKNN.pdf', dpi = 300)












        




#%% CH-index

warnings.filterwarnings("ignore")

sensor = 'Gel'
subject = 'S3'

CH_dict = {'knn':[],'withoutMI':[],'full':[]}

no_of_run = 10

dim_reducer = PCA(n_components = 0.95)
scaler = StandardScaler()


data_full = final_transform(sensor, subject, trial_vector, window_size, feature_vector_list,
                              mi_feature_list, missing_percentage = 0, 
                              imputation_method = 'knn', gp_ls_bounds = (1e-5,1e5),
                              random_state = None)

data = data_full.iloc[:,:-1]
labels = data_full.iloc[:,-1]

data = scaler.fit_transform(data)
projected_full = pd.DataFrame(dim_reducer.fit_transform(data))


print('The number of components is: %s' % (dim_reducer.n_components_))

CH_dict['full'].append(CHScore(projected_full,labels))


sim_seed = 123456



    
data_woMI = final_transform(sensor, subject, trial_vector, window_size, feature_vector_list,
                              mi_feature_list, missing_percentage = 'Gait without MI' , 
                              imputation_method = 'zero', gp_ls_bounds = (1e-5,1e5),
                              random_state = int(sim_seed))

data = data_woMI.iloc[:,:-1]
labels = data_woMI.iloc[:,-1]

data = scaler.fit_transform(data)
projected_woMI = pd.DataFrame(dim_reducer.fit_transform(data))


#print('The number of components is: %s' %(dim_reducer.n_components_))

CH_dict['withoutMI'].append(CHScore(projected_woMI,labels))
    

print('Current data set is %s %s' % (subject, sensor))

print('CH score after zero-imputation is: %s ' % (np.array(CH_dict['withoutMI'])[0]))

print('CH score when data is full: %s ' % (CHScore(projected_full,labels)))



#CH-index for K-NN

sim_seed = 123456

    
data_knn = final_transform(sensor, subject, trial_vector, window_size, feature_vector_list,
                              mi_feature_list, missing_percentage = 0.5 , 
                              imputation_method = 'knn', gp_ls_bounds = (1e-5,1e5),
                              random_state = int(sim_seed))

data = data_knn.iloc[:,:-1]
labels = data_knn.iloc[:,-1]

data = scaler.fit_transform(data)
projected_knn = pd.DataFrame(dim_reducer.fit_transform(data))


#print('The number of components is: %s' %(dim_reducer.n_components_))

CH_dict['knn'].append(CHScore(projected_knn,labels))
    

print('Current data set is %s %s' % (subject, sensor))

print('CH score after k-nn imputation is: %s +- %s' % (np.array(CH_dict['knn']).mean(), 
                                                       np.array(CH_dict['knn']).std()))

print('CH score when data is full: %s ' % (CHScore(projected_full,labels)))



joined_knn = projected_knn.join(labels)
joined_woMI = projected_woMI.join(labels)
joined_full = projected_full.join(labels)


# Plotting:


for i,j in combinations([1,2,3],2):
    
    plt.figure()
    sns.distplot(joined_full.loc[joined_full['data_5'] == i, 0], hist = False,
                 label = 'Class %s' % i, kde_kws={"shade": True}, color = color_kde(i))
    plt.axvline(joined_full.loc[joined_full['data_5'] == i, 0].mean() , 0, 1, linestyle = '--', 
                color = color_mean(i), label = 'Class %s Mean' % i)
    sns.distplot(joined_full.loc[joined_full['data_5'] == j, 0], hist = False, 
                 label = 'Class %s' % j, kde_kws={"shade": True}, color = color_kde(j))
    plt.axvline(joined_full.loc[joined_full['data_5'] == j, 0].mean() , 0, 1,linestyle = '--',
                color = color_mean(j), label = 'Class %s Mean' % j)
    plt.legend()
    plt.xlabel(None)
    sns.despine()
    plt.savefig('Results Figures/%s%sKLclass%sclass%sFullEachComponents.pdf' %
                (subject,sensor,i,j), dpi = 300)
    


for i,j in combinations([1,2,3],2):
    
    plt.figure()
    sns.distplot(joined_knn.loc[joined_knn['data_5'] == i, 0], hist = False,
                 label = 'Class %s' % i, kde_kws={"shade": True}, color = color_kde(i))
    plt.axvline(joined_knn.loc[joined_knn['data_5'] == i, 0].mean() , 0, 1, linestyle = '--', 
                color = color_mean(i), label = 'Class %s Mean' % i)
    sns.distplot(joined_knn.loc[joined_knn['data_5'] == j, 0], hist = False, 
                 label = 'Class %s' % j, kde_kws={"shade": True},color = color_kde(j))
    plt.axvline(joined_knn.loc[joined_knn['data_5'] == j, 0].mean() , 0, 1,linestyle = '--',
                color = color_mean(j), label = 'Class %s Mean' % j)
    plt.legend()
    plt.xlabel(None)
    sns.despine()
    plt.savefig('Results Figures/%s%sKLclass%sclass%sKNNEachComponents.pdf' %
                (subject,sensor,i,j), dpi = 300)
    
    
for i,j in combinations([1,2,3],2):
    
    plt.figure()
    sns.distplot(joined_woMI.loc[joined_woMI['data_5'] == i, 0], hist = False,
                 label = 'Class %s' % i, kde_kws={"shade": True},color = color_kde(i))
    plt.axvline(joined_woMI.loc[joined_woMI['data_5'] == i, 0].mean() , 0, 1, linestyle = '--', 
                color = color_mean(i), label = 'Class %s Mean' % i)
    sns.distplot(joined_woMI.loc[joined_woMI['data_5'] == j, 0], hist = False, 
                 label = 'Class %s' % j, kde_kws={"shade": True},color = color_kde(j))
    plt.axvline(joined_woMI.loc[joined_woMI['data_5'] == j, 0].mean() , 0, 1,linestyle = '--',
                color = color_mean(j), label = 'Class %s Mean' % j)
    plt.legend()
    plt.xlabel(None)
    sns.despine()
    plt.savefig('Results Figures/%s%sKLclass%sclass%sGaitwoMIEachComponents.pdf' %
                (subject,sensor,i,j), dpi = 300)








