import pip
import time
import random
import numpy as np
import pandas as pd

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

    numerical_data_starting_index = no_of_time_features
    categorical_data_starting_index = numerical_data_starting_index + no_of_numerical_features
    mvc_data_starting_index = categorical_data_starting_index + no_of_categorical_features

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
        'numerical_data_starting_index': numerical_data_starting_index,
        'categorical_data_starting_index': categorical_data_starting_index,
        'mvc_data_starting_index': mvc_data_starting_index,
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
        data = np.nan_to_num(F['numerical'])

    if info['no_of_categorical_features'] > 0:
        data = F['CAT'].fillna('nan').values if len(data) == 0 else np.concatenate((data, F['CAT'].fillna('nan').values), axis=1)

    if info['no_of_mvc_features'] > 0:
        data = F['MV'].fillna('nan').values if len(data) == 0 else np.concatenate((data, F['MV'].fillna('nan').values), axis=1)

    print('data.shape: {}\n'.format(data.shape))
    return data

def split_data_by_type(data, info):
    print('\nsplit_data_by_type')

    time_data = np.array([]) if info['no_of_time_features'] == 0 else data[:,:info['numerical_data_starting_index']]
    numerical_data = np.array([]) if info['no_of_numerical_features'] == 0 else \
                    data[:,info['numerical_data_starting_index']:info['categorical_data_starting_index']]
    categorical_data = np.array([]) if info['no_of_categorical_features'] == 0 else \
                    data[:,info['categorical_data_starting_index']:info['mvc_data_starting_index']]
    mvc_data = np.array([]) if info['no_of_mvc_features'] == 0 else \
                    data[:,info['mvc_data_starting_index']:]

    print('time_data.shape :{}'.format(time_data.shape))
    print('numerical_data.shape :{}'.format(numerical_data.shape))
    print('categorical_data.shape :{}'.format(categorical_data.shape))
    print('mvc_data.shape :{}\n'.format(mvc_data.shape))
    return time_data, numerical_data, categorical_data, mvc_data

def subtract_min_time(time_data):
    print('\nsubtract_min_time')
    print('time_data.shape: {}'.format(time_data.shape))
    # result = np.apply_along_axis(
    #     lambda x: x.astype(float) - np.min(x[np.nonzero(x.astype(float))[0]]), 
    #     0, 
    #     time_data
    # )
    no_of_rows, no_of_cols = time_data.shape
    result = []
    for i in range(no_of_cols):
        time_col = time_data[:,i].astype(float)
        non_zero_indices = np.nonzero(time_col)[0]
        difference = time_col - np.min(time_col[non_zero_indices])
        result.append(difference)
    result = np.array(result).T
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
                    difference = difference.reshape((-1, 1))
                    result = difference if len(result) == 0 else np.concatenate((result, difference), axis=1)
    print('result.shape: {}\n'.format(result.shape)) 
    return result

def has_sufficient_time(dataset_budget_threshold, info):
    return info['dataset_time_spent'] < info['time_budget'] * dataset_budget_threshold
