import pip
def pip_install(package):
    pip.main(['install', package])

import time
import random
import numpy as np
import pandas as pd

pip_install('category_encoders')

from category_encoders import HashingEncoder

def pip_install(package):
    pip.main(['install', package])

def extract(datainfo, timeinfo):
    time_budget = datainfo['time_budget']
    no_of_time_features = datainfo['loaded_feat_types'][0]
    no_of_numerical_features = datainfo['loaded_feat_types'][1]
    no_of_categorical_features = datainfo['loaded_feat_types'][2]
    no_of_mvc_features = datainfo['loaded_feat_types'][3]
    total_no_of_features = no_of_time_features + no_of_numerical_features + \
                        no_of_categorical_features + no_of_mvc_features

    current_time = time.time() 
    overall_time_spent = current_time - timeinfo[0]
    dataset_time_spent = current_time- timeinfo[1]

    return {
        'time_budget': time_budget,
        'no_of_time_features': no_of_time_features,
        'no_of_numerical_features': no_of_numerical_features,
        'no_of_categorical_features': no_of_categorical_features,
        'no_of_mvc_features': no_of_mvc_features,
        'total_no_of_features': total_no_of_features,
        'overall_time_spent': overall_time_spent,
        'dataset_time_spent': dataset_time_spent
    }

def print_data_info(info):
    print('\nDataset time budget: {0:f} seconds'.format(info['time_budget']))
    print('No. of time features: {0:d}'.format(info['no_of_time_features']))
    print('No. of numerical features: {0:d}'.format(info['no_of_numerical_features']))
    print('No. of categorical features: {0:d}'.format(info['no_of_categorical_features']))
    print('No. of mvc features: {0:d}\n'.format(info['no_of_mvc_features']))

def print_time_info(info):
    print('\nOverall time spent: {0:5.2f} seconds'.format(info['overall_time_spent']))
    print('Dataset time spent: {0:5.2f} seconds\n'.format(info['dataset_time_spent'])) 

def get_data(F, info):
    print('\nget_data')

    data = np.array([])
    if info['no_of_time_features'] > 0 or info['no_of_numerical_features'] > 0:
        data = F['numerical']

    if info['no_of_categorical_features'] > 0:
        data = F['CAT'].values if len(data) == 0 else np.concatenate((data, F['CAT'].values), axis=1)

    if info['no_of_mvc_features'] > 0:
        data = F['MV'].values if len(data) == 0 else np.concatenate((data, F['MV'].values), axis=1)

    print('data.shape: {}\n'.format(data.shape))
    return data

def split_data_by_type(data, info):
    print('\nsplit_data_by_type')

    numerical_data_starting_index = info['no_of_time_features']
    categorical_data_starting_index = numerical_data_starting_index + info['no_of_numerical_features']
    mvc_data_starting_index = categorical_data_starting_index + info['no_of_categorical_features']

    time_data = np.array([]) if info['no_of_time_features'] == 0 else data[:,:numerical_data_starting_index]
    numerical_data = np.array([]) if info['no_of_numerical_features'] == 0 else \
                    data[:,numerical_data_starting_index:categorical_data_starting_index]
    categorical_data = np.array([]) if info['no_of_categorical_features'] == 0 else \
                    data[:,categorical_data_starting_index:mvc_data_starting_index]
    mvc_data = np.array([]) if info['no_of_mvc_features'] == 0 else \
                    data[:,mvc_data_starting_index:]

    print('time_data.shape :{}'.format(time_data.shape))
    print('numerical_data.shape :{}'.format(numerical_data.shape))
    print('categorical_data.shape :{}'.format(categorical_data.shape))
    print('mvc_data.shape :{}\n'.format(mvc_data.shape))
    return time_data, numerical_data, categorical_data, mvc_data

def convert_nan(data):
    return np.nan_to_num(time_data)

def subtract_min_time(time_data):
    print('\nsubtract_min_time')
    print('time_data.shape: {}'.format(time_data.shape))
    result = np.apply_along_axis(
        lambda x: x.astype(float) - np.min(x[np.nonzero(x)[0]]), 
        0, 
        time_data
    )
    print('result.shape: {}\n'.format(result.shape)) 
    return result

def difference_between_time_columns(time_data):
    print('\ndifference_between_time_columns')
    no_of_rows, no_of_cols = time_data.shape
    print('time_data.shape: {}'.format((no_of_rows, no_of_cols)))
    result = np.array([])
    for i in range(no_of_cols):
            for j in range(i+1, no_of_cols):
                if len(np.nonzero(time_data[:,i])) > 0 and len(np.nonzero(time_data[:,j])) > 0:
                    difference = time_data[:,i] - time_data[:,j]
                    result = difference if len(result) == 0 else np.concatenate((result, difference), axis=1)
    print('result.shape: {}\n'.format(result.shape)) 
    return result

def compute_woe(categorical_or_mvc_data, labels=None, woe_map={}):
    print('\ncompute_woe')
    no_of_rows, no_of_cols = categorical_or_mvc_data.shape
    print('categorical_or_mvc_data.shape: {}'.format((no_of_rows, no_of_cols)))
    
    result = np.array([])
    if labels is None and not woe_map: # predict
        for i in range(no_of_cols):
            d0 = pd.DataFrame({'X': categorical_or_mvc_data[:,i]})
            d1 = d0.join(woe_map[i], on='X')[['WOE']].values
            result = d1 if len(result) == 0 else np.concatenate((result, d1), axis=1)
            del d0
    else: #fit
        for i in range(no_of_cols):
            d0 = pd.DataFrame({'X': categorical_or_mvc_data[:,i], 'Y': labels.ravel()})
            d1 = d0.groupby('X',as_index=True)
            d2 = pd.DataFrame({},index=[])
            d2['COUNT'] = d1.count().Y
            d2['EVENT'] = d1.sum().Y
            d2['NONEVENT'] = d2.COUNT - d2.EVENT
            d2['DIST_EVENT'] = d2.EVENT/d2.sum().EVENT
            d2['DIST_NON_EVENT'] = d2.NONEVENT/d2.sum().NONEVENT
            d2['WOE'] = np.log(d2.DIST_EVENT/d2.DIST_NON_EVENT)
            # d2['IV'] = (d2.DIST_EVENT - d2.DIST_NON_EVENT) * d2.WOE
            d2 = d2[['COUNT','WOE']] 
            d2 = d2.replace([np.inf, -np.inf], 0)
            # d2.IV = d2.IV.sum()

            woe_map[i] = d2
            d3 = d0.join(d2, on='X')[['WOE']].values
            result = d3 if len(result) == 0 else np.concatenate((result, d3), axis=1)   
            del d0
            del d1
    print('result.shape: {}\n'.format(result.shape)) 
    return result, woe_map

def hash(categorical_or_mvc_data, labels=None, encoder=None):
    print('\nhash')
    result = np.array([])
    if labels is None and encoder is not None: # predict
        result = encoder.transform(categorical_or_mvc_data)
    else: #fit
        encoder = HashingEncoder()
        result = encoder.fit_transform(categorical_or_mvc_data, labels)

    print('result.shape: {}\n'.format(result.shape)) 
    return result, encoder

def has_sufficient_time(dataset_budget_threshold, info):
    return info['dataset_time_spent'] < info['time_budget'] * dataset_budget_threshold
