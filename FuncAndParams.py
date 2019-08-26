
# Author: Nghi Le

# Basic:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings

# Signal processing:

from scipy import signal
import scipy as sp

# Others:

from sklearn.metrics import calinski_harabasz_score as CHScore
from sklearn import svm
from sklearn.pipeline import make_pipeline
from joblib import Memory
from tempfile import mkdtemp
from shutil import rmtree
import pickle
from scipy.stats import ttest_ind
from numpy.linalg import norm
from itertools import combinations
from scipy import stats
from scipy.special import kl_div
from time import time

# Model selection:

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# Imputation:

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, MissingIndicator
from missingpy import KNNImputer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from autoimpute.imputations import SingleImputer, MultipleImputer 


cachedir = mkdtemp('Cache')
memory = Memory(location=cachedir, verbose=0)


# Save and Load Experimental results:

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


#### Project steps:

# 1. Filter/preprocessing the signal. (Simulate missingness.)
# 2. (Imputation?)
# 3. Feature engineering and extract. Picking features and such.
# 4. Scaling and PCA. Put this into the pipeline.
# 5. Classification and cross validation. Hyperparameter selection. Put the pipe
# into GridSearchCV function.

#### Pre-processing / filtering:

def preprocess(data, filter_order = 8, cutoff_high = 20, cutoff_low = 450, sampling_rate = 1000):
    
    signals = data.iloc[:,:-1]
    labels = data.iloc[:,-1]
    
    # Mean subtraction:
    signals = signals - signals.mean()
    
    # Bandpass filter:
    normalized_freq = np.array([cutoff_high, cutoff_low]) / (sampling_rate / 2)
    b, a = sp.signal.butter(filter_order, normalized_freq , btype = 'bandpass')
    
    # Power-line noise removal:
    notch_freq = np.array([48 , 52]) / (sampling_rate / 2)
    e, f = sp.signal.butter(3, notch_freq, btype = 'bandstop')
    
    # Apply filter forward and backward:
    signals = signals.apply(lambda x: sp.signal.filtfilt(b, a, x))
    signals = signals.apply(lambda x: sp.signal.filtfilt(e, f, x))
    
    return(signals.join(labels))
    


#### Simulate missingess

# Channel 1: Quadriceps - data_1
# Channel 2: Hamstring - data_2
# Channel 3: Tibialis Anterior - data_3
# Channel 4: Triceps Surae - data_4

# HS = Heel-strike ( label 1 )
# RS = Rest (label 2)
# TO = Toe-off (label 3)

# Missing data assumptions:

# When we do a heel strike, the Hamstring and the Triceps Surae will most
# likely has missing data. For a standard pant, the area around the Hamstring
# is usuallly more tight so it probably have less missing data than the hamstring.

# When we do a toe-off, the quadriceps and the triceps surae will most likely
# have missing data. Also, the area around the upper part of the leg will probably
# have more missing data for the same reason as above.

# Also, data will be missing randomly around the whole data set due to
# general looseness of the frabric and other possible noises. But 
# of course not as big a section or as consistent as missingness
# due to gait.

# The data we will take out:

# 1. For Triceps Surae, channel 4, the data will be missing the whole time the
# label is HS. When halfway through the rest label the data is back for
# the entire duration of the TO label and about more than halfway through
# the next RS period.

# 2. We can reasonably believe the hamstrings channel will have similar
# missingness patterns, but the missiness period are less.

# 3. For Tibialis anterior the reasonable assumption will be that the missingness pattern
# is completely opposite to the missingness pattern of the Triceps Surae.

# 4. The quadriceps will then be completely opposite to the hamstrings.

# 5. Randomly take out some samples around the data set to simulate further
# noises and missingness.

# If we take out the data according to the labels then the code can generalize across
# different dataset.

    
# Missing data simulation function:

# Since the data always starts with the first heel strike, we can assume the missing
# pattern is going to be kind of the same.

# Separate each rest section into a separate dictionary entry:

def rest_data_processing(data):
    
    # Get the index of rest-labeled data
    rs_index = data.loc[data['data_5'] == 2].index
    index_list = []
    prev_index = rs_index[0] - 1
    
    # Create dictionary to store different sections of the rest signal.
    index_dict = {}
    section_count = 0
    
    # Separate the index of the rest signal into different sections.
    for i in rs_index:
        
        index_list.append(i)
        
        if i - 1 != prev_index:
            section_count += 1
            index_dict['rest_section_' + str(section_count)] = index_list[:-1]
            index_list = []
            index_list.append(i)
            
        if i == rs_index[-1]:
            section_count += 1
            index_dict['rest_section_' + str(section_count)] = index_list
            index_list = []
            
        prev_index = i
    
    return(index_dict)
    

# Determining which index to take out for rest labeled data:

def take_out_index_RS(data):
    
    index_dict = rest_data_processing(data)
    take_out_index_RS = []
    
    # Does not include the last rest section because it is a very short one that
    # probably does not have much missing data.
    for i in range(1, len(index_dict.keys())):
        index = index_dict['rest_section_' + str(i)]
        mid_point = int(len(index) / 2)
        
        if i % 2 == 0:
            take_out_index_RS.extend(index[mid_point:])
        else:
            take_out_index_RS.extend(index[:mid_point])
    
    return(take_out_index_RS)


# Take out the data according to assumptions:

def take_out_data(data):
    
    hs_index = list(data.loc[data['data_5'] == 1].index)
    rs_index = list(data.loc[data['data_5'] == 2].index)
    to_index = list(data.loc[data['data_5'] == 3].index)
    take_out_index_rs = take_out_index_RS(data)
    final_take_out_index = []
    
    for channel in [1,2,3,4]:
        
        # Determine the data points to take out for each channel
        if channel == 4: # Triceps Surae
            final_take_out_index =  take_out_index_rs + hs_index
        elif channel == 3: # Tibialis Anterior
            # Opposite patterns with Triceps surae
            final_take_out_index = list(set(rs_index) - set(take_out_index_rs)) + to_index
        elif channel == 2: # Hamstring
            # Same pattern with Triceps surae for now. Maybe add some noises in
            # later.
            final_take_out_index =  take_out_index_rs + hs_index
        elif channel == 1: # Quadriceps
            final_take_out_index = list(set(rs_index) - set(take_out_index_rs)) + to_index
            
        # Take out the data point.
        data.iloc[final_take_out_index, channel - 1] = np.nan
    
    return(data)

# Take out data randomly: This will create a situation where the data is MCAR.
    
def take_out_random(data, percentage, random_state = None):
    
    for channel in [1,2,3,4]:
        
        # Ensure the patterns is different every channel:
        if random_state != None:
            random.seed(random_state + channel)
            
        index = random.sample(list(data.index), round(float(percentage) * len(data)))
        
        data.iloc[index,channel - 1] = np.nan
        
    return(data)
    
# Get indexes of missing values of each columns:
    
def nan_index(vector):
    
    nan_index = np.array(vector.index[vector.apply(np.isnan)])
    
    return(pd.DataFrame(nan_index))
    
    
def non_nan_index(vector):
    
    nan_index = vector.index[vector.apply(np.isnan)]
    
    non_nan_ind = np.array( list(set(vector.index) - set(nan_index)))
    
    return(pd.DataFrame(non_nan_ind))
    

#### Imputation for missing data:

# 1. Zero imputation:

def zero_impute(data):
    
    imputer = SimpleImputer(missing_values = np.nan, strategy = 'constant',
                            fill_value = 0)
    
    imputed_df = pd.DataFrame(imputer.fit_transform(data))
    
    # Get the column names back as the imputer dropped this information:
    imputed_df.columns = data.columns
    
    return(imputed_df)
    

# 2. Mean imputation: Using column mean. Not row mean.

def mean_impute(data):
    
    imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
    
    imputed_df = pd.DataFrame(imputer.fit_transform(data))
    
    # Get the column names back as the imputer dropped this information:
    imputed_df.columns = data.columns
    
    return(imputed_df)
    
# 3. PMM: This will return a point-estimate single regression imputation, not a full multiple
# imputation implementation. For full implementation details look at Azur et al. 2011 and:
# https://scikit-learn.org/stable/modules/impute.html#multiple-vs-single-imputation
# Cross-sectional.

def pmm_impute(data, n_impute = 3):
    
    signals = data.iloc[:, :-1]
    labels = data.iloc[:,-1]
    
    # PMM default number of neighbors is 5:
    imputer = MultipleImputer(n = n_impute, strategy = 'pmm', return_list = True,
                              imp_kwgs = {'tune' : 1000, 'sample' : 1000, 'neighbors' : 5})
    
    data_sets = imputer.fit_transform(signals)
    
    signals = avg_data_sets(data_sets)
    
    return(signals.join(labels))


# 4. K-NN imputation: from missingpy package: https://pypi.org/project/missingpy/
# Note: This is a cross sectional method. So the distance metric used does not exploits
# temporal relations. It is found in Jerez et al. 2010 that K-NN has similar performance
# to other cross sectional methods such as MLP and SOM.

def knn_impute(data, n_neighbors = 3):
    
    imputer = KNNImputer(n_neighbors = n_neighbors, missing_values = np.nan,
                         weights = 'distance')
    
    imputed_df = pd.DataFrame(imputer.fit_transform(data))
    
    imputed_df.columns = data.columns
    
    return(imputed_df)
    
# 5. Single-Task Gaussian Process imputer: temporal method. Utilizing distance
# across time. The whole signal is used for regression.
    
def gp_impute(data, ls_bounds, n_restarts = 0,random_state = None):
    
    kernel = RBF(length_scale_bounds = ls_bounds) + WhiteKernel()
    
    signals = data.iloc[:, :-1]
    labels = data.iloc[:,-1]
    
    gpr = GaussianProcessRegressor(kernel = kernel, random_state = random_state,
                                   n_restarts_optimizer = n_restarts)
    
    # Imputing column by column:
    for i in signals.columns:
        
        # Get the indexes for missing values and non-missing values:
        nan_indexes = nan_index(signals[i])
        non_nan_indexes = non_nan_index(signals[i])
        
        # Gaussian Process regression:
        gpr.fit(non_nan_indexes, signals.loc[ non_nan_indexes[0] ,i])
        
        # Get the imputated values at the indexes where data is missing:
        impute_values = pd.DataFrame(gpr.predict(nan_indexes))
        
        # Fill in the values:
        impute_values.index = nan_indexes[0]
        signals[i] = signals[i].fillna(value = impute_values[0])
        
        #print(gpr.kernel_)
    
    return(signals.join(labels))
    

# Impute wrapper:
    
def impute(data, method, gp_ls_bounds = (1e-5,1e5)):
    
    if method == 'zero':
        data = zero_impute(data)
    elif method == 'mean':
        data = mean_impute(data)
    elif method == 'knn':
        data = knn_impute(data)
    elif method == 'gp':
        data = gp_impute(data, ls_bounds = gp_ls_bounds)
    elif method == 'pmm':
        data = pmm_impute(data)
    
    return(data)

# Get missingness indicators:
    
def get_indicators(data):
    
    indicator = MissingIndicator(missing_values = np.nan, features = 'all')
    mask_data = pd.DataFrame(indicator.fit_transform(data.iloc[:,:-1]))
    
    # Rename some columns:
    mask_data.columns = mask_data.columns + 1
    mask_data = mask_data.add_prefix('ind_')
    
    return(mask_data)

# This function is to average the imputed values from Predictive Mean Matching.
# Pass in a list of data frames and return a single data frame with the average
# imputed values:
    
def avg_data_sets(data_frames):
    
    total_sum = pd.DataFrame(data_frames[0][1])
    
    for i in range(0, len(data_frames)):
        
        if  i != (len(data_frames) - 1):
            total_sum = total_sum + pd.DataFrame(data_frames[i + 1][1])
            
    return(total_sum / len(data_frames))
    
    
# This function averages the dataframes in a dictionary:

def avg_dict_df(dictionary):
    
    total_sum = dictionary[0]
    
    for i in range(len(dictionary)):
        
        if i != (len(dictionary) - 1):
            total_sum = total_sum + dictionary[i + 1]
            
    return(total_sum / len(dictionary))
    
# Get std for dictionary of results:

def std_dict_df(df_acc_over_rates, imputation_methods, missing_rates = [0,1,2,3]):
    
    std_dict = {}
    acc_dict = {}
    
    for imp in imputation_methods:
        std_dict[imp] = []
        for rates in missing_rates:
            acc_dict[imp] = []
            for i in range(len(df_acc_over_rates)):
                acc_dict[imp].append(df_acc_over_rates[i].loc[rates,imp])
                
            std_dict[imp].append(np.array(acc_dict[imp]).std())
            
    std_df = pd.DataFrame.from_dict(std_dict) 
    std_df.index = missing_rates
    
    return(std_df)
    
# Get summary table of results.
        
def combine_acc_std(acc,std, missing_rates = [0, 0.25, 0.5, 0.75]):
    
    acc = (acc.round(4) * 100).round(2).applymap(str)
    std = (std.round(4) * 100).round(2).applymap(str)

    
    combine = pd.DataFrame(acc.applymap(str) + ' $ \pm $ '+ std.applymap(str))
    
    
    missing_rates = [str(int(i * 100)) + '$\%$' if isinstance(i, (float,int)) 
                    else i for i in missing_rates]
    
    combine.index = missing_rates
    
    return(combine)
    
# Get GP bounds from simulation results
def get_gp_bounds(subject, sensor, missing_rates):
    
    if isinstance(missing_rates, str):
        results = load_obj('Gait GP Bounds simulations results over %s runs %s %s' 
                           % (20, subject, sensor))
    else:
        results = load_obj('GP Bounds simulations results over %s runs %s %s' 
                      % (20, subject, sensor))
    

    results = avg_dict_df(results)
    
    optimal = results.loc[missing_rates,:].astype('float').idxmax()
    
    print('The average results over 20 simulations of GP bounds is:')
    print()
    print(results)
    print()
    print('The optimal GP bounds for %s %s at %s missing rate is:' %
          (subject,sensor,missing_rates))
    print()
    print(optimal)
    print()
    
    return(optimal)


def color_kde(i):
    if i == 1:
        return('tab:blue')
    elif i == 2:
        return('tab:orange')
    elif i == 3:
        return('tab:green')
        
def color_mean(i):
    if i == 1:
        return('cornflowerblue')
    elif i == 2:
        return('orange')
    elif i == 3:
        return('limegreen')
    
    

#### Feature extraction: Phinyomark et al. 2012 reccomends MAV, WL, WAMP, 4th-order AR:

# 1. Waveform length:

def wl(vector): 
    
    a = abs(vector.diff())
    
    # Adjusted by size:
    return(a.sum() / a.count()) 


# 2. Mean absolute value:
    
def mav(vector):
    
    return(abs(vector).mean())
    
# 3. Willison amplitude: Assuming our amplitude is in volt. 0.015 means 15mV
# which is an arbitrary threshold. This feature has a filtering effect on White noise: Phinyomark et al. 2008
    
def wamp(vector, threshold = 0.025):  
    
    a = abs(vector.diff())
    
    a[a >= threshold] = 1
    a[a < threshold] = 0
    
    return(a.sum())
    
#### Missingness indicators features: according to Lipton et al. 2016
    
# 4. Mean of the indicators: capture the frequency of missing data. 
    
def mi_mean(mi):
    
    return(mi.mean())
    
# 5. Standard deviation of the indicators:
    
def mi_std(mi):
    
    return(mi.std())
    

# 6. Capture relative timing of the missing values: This is a hand-engineered
# feature.
    
def mi_timing(mi):
    
    mid_point = len(mi) // 2
    
    # Conversion of True and False to 1 and Nan respectively:
    
    timing_data = mi[mi == True]
    
    for i in range(0, len(timing_data.columns)):    
        timing_data.iloc[np.where(mi.iloc[:,i])[0], i] = np.where(mi.iloc[:,i])[0]
    
    relative_timing = (timing_data - mid_point) / mid_point
    
    return(relative_timing.sum())
    

# Feature extraction:

def get_feature(data, window_size, feature_function):
    
    # Apply waveform_length() to every 200 samples
    feature_data = data.groupby(data.index // window_size).apply(lambda x: feature_function(x))
    
    feature_data = feature_data.add_suffix('_' + feature_function.__name__)
    
    # Drop last row because the window size is only 2ms.
    return(feature_data[:-1]) 
    
# Get label function:

def get_label(data, window_size):
    
    # Get the mode by time-window. Basically implementing majority-voting:
    label_data = data.groupby(data.index // window_size).agg(pd.Series.mode) 
    
    return(label_data[:-1])
    
# Transform missing indicators into feature sets:
    
def transform_mi(mi_ind, window_size, feature_function_list):
    
    # Create empty dataframe first
    transformed_df = pd.DataFrame(index = range(0, len(mi_ind) // window_size))
    
    # Loop over every feature and join them into a dataframe:
    for function in feature_function_list:
        feature_data = get_feature(mi_ind, window_size, function)
        transformed_df = transformed_df.join(feature_data)
    
    return(transformed_df)
    
# Transform each dataset into set of determined features:
    
def transform_data(data, mi_ind, window_size, signal_feature_list, mi_feature_list):
    
    data_signal = data.iloc[:,:-1]
    data_labels = data.iloc[:,-1]
    
    # Create empty dataframe first
    transformed_df = pd.DataFrame(index = range(0, len(data_signal) // window_size))
    
    # Loop over every feature and join them into a dataframe:
    for function in signal_feature_list:
        signal_feature_data = get_feature(data_signal, window_size, function)
        transformed_df = transformed_df.join(signal_feature_data)
    
    if mi_ind is not None:
        mi_feature_data = transform_mi(mi_ind, window_size = window_size, 
                                       feature_function_list = mi_feature_list)
        # Attach the indicators feature to data frame.
        transformed_df = transformed_df.join(mi_feature_data)
    
    # Get the labels:
    labels = get_label(data_labels, window_size)
    
    # Attach the labels to data frame together:
    transformed_df = transformed_df.join(labels)
    
    return(transformed_df)

# Put all samples from every subject and every trial into one dataset:

def final_transform(reader, subject, trial_vector, window, signal_features, 
                    mi_ind_features, missing_percentage, gp_ls_bounds,imputation_method = 'zero', 
                    random_state = None):
    
    final_df = pd.DataFrame()
    
    #Default setting for missingness indicators, if None then the transform_data
    #function does not use mi_ind:
    mi_ind = None
    mi_assumptions = None
    
    for t in trial_vector:
        
        # Filename:
        name = subject + reader + '_' + t + '.txt'
        
        with open(name) as file:
            data = pd.read_csv(file, header = 0)
            
        # Filtering:
        data = preprocess(data, 8, 20, 450)
        
        if missing_percentage == 'Gait without MI':
            mi_assumptions = True
            mi_features = False
        elif missing_percentage == 'Gait with MI':
            mi_assumptions = True
            mi_features = True
        
        # Toggle simulation according to assumptions:
        if mi_assumptions == True:
            
            # Take out data according to assumptions:
            data = take_out_data(data)
            
            #Toggle of whether we use indicators based features or not.
            if mi_features == True:
                # Get missing indicators and feature extract:
                mi_ind = get_indicators(data)
                
            # Imputation:
            data = impute(data , imputation_method, gp_ls_bounds = gp_ls_bounds)
        
        # Toggle simulation of MCAR missing data. At this point, this setting
        # will be mutually exclusive with the mi_assumptions setting. Further
        # exploration with both type of missing data together will be explored
        # in the future.
        if missing_percentage != 0 and isinstance(missing_percentage, float):
            
            # Ensure every trial has a different missing patterns:
            if random_state != None:
                random_state = random_state + int(list(t)[1]) * 10
            
            # Simulate missing data:
            data = take_out_random(data, missing_percentage,
                                   random_state = random_state)
            
            # Imputation:
            data = impute(data , imputation_method,  gp_ls_bounds = gp_ls_bounds)
        
        # Transform each dataset into set of features:
        data_frame = transform_data(data, mi_ind, window, signal_features, mi_ind_features)
        
        # Concatenate the dataset across trials into one:
        final_df = final_df.append(data_frame)
        
    return(final_df.reset_index(drop = True))
    
    

# Classification and Cross-validation function:

def cv_predict_report ( data, n_splits, test_size, param_grid, scores, dim_reduction = 'PCA',
                       random_state = None, verbose = True):
    
    # Implement verbose option:
    verboseprint = print if verbose else lambda *a, **k: None
    
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    # Using stratified split due to class imbalances:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                        shuffle = True,
                                                        stratify = y,
                                                        random_state=random_state)
    
    # StratifiedShuffleSplit for 10-fold cross validation:
    cv = StratifiedShuffleSplit(n_splits = n_splits, test_size = test_size, 
                                random_state = random_state * 7 )

    if dim_reduction == 'PCA':
        dim_red = PCA(n_components = 0.95)
    elif dim_reduction == 'LDA':
        dim_red = LinearDiscriminantAnalysis()
        
    for score in scores:
        verboseprint()
        verboseprint("# Tuning hyper-parameters for %s" % score)
        verboseprint()
        
        # Basically, this pipeline first standardize the data to a standard normal distribution 
        # with mean 0 and variance 1, then performs PCA, and then pass the normalized 
        # data through a 10-fold cross-validating classifier with grid search for hyperparameters:
        
        # Note that sklearn imputation also implements fit-transform, so can be used
        # as part of the pipeline. Obviously if we uses it like this, it will
        # be imputation after feature extraction. If we want to impute before
        # feature extraction, don't put it in the pipeline.
        
        pipe = make_pipeline(StandardScaler(), dim_red, svm.SVC(kernel = 'rbf'),
                             memory = memory)
        
        
        # GridSearch for cross-validation:
        clf = GridSearchCV(pipe, param_grid = param_grid, cv = cv, refit = True, scoring = score)
        
        clf.fit(X_train, y_train)
        
        #if dim_reduction == 'PCA':
            #verboseprint("Number of principal components:")
            #verboseprint()
            # Accessing number of components on best estimator:
            #verboseprint(clf.best_estimator_.named_steps['pca'].n_components_)
            #verboseprint()
            #verboseprint("Variance ratio explained by %s principal components:" 
            #             % (clf.best_estimator_.named_steps['pca'].n_components_))
            #verboseprint()
            #verboseprint(clf.best_estimator_.named_steps['pca'].explained_variance_ratio_.sum())
        #elif dim_reduction == 'LDA':
            #verboseprint("Variance ratio explained by principal components:" )
            #verboseprint()
            #verboseprint(clf.best_estimator_.named_steps['lineardiscriminantanalysis'].explained_variance_ratio_)
        verboseprint()
        verboseprint("Best parameters set found on development set:")
        verboseprint()
        verboseprint(clf.best_params_)
        verboseprint()
        verboseprint("With the best mean scores of:")
        verboseprint()
        verboseprint(clf.best_score_)
        verboseprint()
        verboseprint("Grid scores on development set:")
        verboseprint()
        
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            verboseprint("%0.3f (+/-%0.03f) for %r"
                         % (mean, std * 2, params))
        verboseprint()
    
        verboseprint("Detailed classification report:")
        verboseprint()
        verboseprint("The model is trained on the full development set.")
        verboseprint()
        verboseprint("The scores are computed on the full evaluation set.")
        verboseprint()
        y_true, y_pred = y_test, clf.predict(X_test)
        #verboseprint(metrics.classification_report(y_true, y_pred))
        verboseprint()
        verboseprint('The accuracy on the test set is:')
        verboseprint()
        verboseprint(metrics.accuracy_score(y_true,y_pred))
        verboseprint()
        verboseprint('The balanced accuracy on the test set is:')
        verboseprint()
        verboseprint(metrics.balanced_accuracy_score(y_true,y_pred))
        #verboseprint('Weighted F1 score is:')
        #verboseprint()
        #verboseprint(metrics.f1_score(y_true, y_pred, average = 'weighted'))
        
    return(metrics.balanced_accuracy_score(y_true,y_pred))

# Scaling function:
    
def Scale(data):
    
    scaler = StandardScaler()
    
    signals = data.iloc[:,:-1]
    labels = data.iloc[:,-1]
    
    signals = pd.DataFrame(scaler.fit_transform(signals))
    
    signals.columns = data.iloc[:,:-1].columns
    
    return(signals.join(labels))    
    
# Get the total difference of distance between the means of each group
    
def SumDist(data, scaling = False):
    
    if scaling == True:
        data = Scale(data)
        
    group_means = data.groupby('data_5').mean()
    
    dist_list = []
    
    for i,j in combinations([0,1,2],2):
        
        dist = norm(group_means.iloc[i,:] - group_means.iloc[j,:])
        
        dist_list.append(dist)
        
        print('Distance from group %s to group %s is: %s' % 
              (i + 1, j + 1, dist))
        
    sum_dist = pd.DataFrame(dist_list).sum().values[0]
    
    print()
    print('Total difference between the distances of the groups is: %s' 
          % (sum_dist))
    
    return(sum_dist)

    
    
    
    
    
    
#### Parameters: This is important

trial_vector = ['t1','t2','t3', 't4', 't5']
window_size = 200

# List of features to be extracted:
feature_vector_list = [wl, mav, wamp]
mi_feature_list = [mi_mean, mi_timing]

# Hyperparameter search space:
#C_range = [0.001,0.01,0.1,1,10,100,1000, 10000000000000]
#gamma_range = [0.001,0.01,0.1,1,10,100,1000,10000000000000]

C_range = [0.001, 0.01, 0.1, 1 ,10, 100, 1000]
gamma_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

param_grid = {'svc__gamma':gamma_range,'svc__C':C_range}

# Scoring setting to evaluate hyperparameters in GridSearchCV:

# Macro-averaging should be used to ensure the model do well with even the minority
# class, not just the majority class.

#scores = ['f1_macro']

# Balanced accuracy can also work to give more weights to the minority class.
# Defined as the macro-average of recall scores per class 

scores = ['balanced_accuracy']

#scores = ['accuracy']


#%% Getting the final dataset example:

# Separate data sets for each subject to document for natural differences between
# subjects.

# Please run this while in the same folder as the dataset.
# Random-state is controlling how the missing data is taken out.
# Indicators_features controls whether to get features calculated from missingness
# indicators or not

#%% Full data with no missingness:
gel_data_s4 = final_transform('Gel', 'S4', trial_vector, window_size, feature_vector_list,
                              mi_feature_list, missing_percentage = 0 , mi_assumptions = False, 
                              imputation_method = 'zero', 
                              random_state = None)

#%% Data with 25% missing at random:
gel_data_s4 = final_transform('Gel', 'S4', trial_vector, window_size, feature_vector_list,
                              mi_feature_list, missing_percentage = 0.25 , mi_assumptions = False, 
                              imputation_method = 'zero', 
                              random_state = None)

#%% Data with missingness according to assumptions with out randomness:
# The missing percentage is automatically set to 0 when mi_assumptions is True.
gel_data_s4 = final_transform('Gel', 'S4', trial_vector, window_size, feature_vector_list,
                              mi_feature_list, missing_percentage = 0.25 , mi_assumptions = True, 
                              imputation_method = 'zero', 
                              random_state = None)


#%% Test run:
cv_predict_report(gel_data_s4, 10, 0.3, param_grid, scores, verbose = True)































































