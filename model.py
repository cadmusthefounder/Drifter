import pickle
import time
import random
from os.path import isfile
import numpy as np
import pandas as pd

from utils import pip_install

pip_install('hyperopt')
pip_install('lightgbm')

from lightgbm import LGBMClassifier
from hyperopt import hp
from hyperopt.pyll.base import scope
from hyperparameters_tuner import HyperparametersTuner

class Model:

    def __init__(self, datainfo, timeinfo):
        info = self._extract(datainfo, timeinfo)
        self._print_data_info(info)
        self._print_time_info(info)
        
        self._max_training_data = 300000
        self._training_data = np.array([])
        self._training_labels = np.array([])
        self._classifier = None
        self._classifier_class = LGBMClassifier
        self._fixed_hyperparameters = {
            'learning_rate': 0.01, 
            'n_estimators': 600, 
            'max_depth': 11, 
            'num_leaves': 110, 
            'max_bin': 150,
            'scale_pos_weight': 1,
            'feature_fraction': 0.6350762584583878, 
            'bagging_fraction': 0.6991186365033116, 
            'bagging_freq': 6, 
            'boosting_type': 'gbdt', 
            'objective': 'binary', 
            'metric': 'auc'
        }
        self._search_space = {
            'n_estimators': scope.int(hp.quniform('n_estimators', 500, 700, 10)), 
            'max_depth': scope.int(hp.quniform('max_depth', 7, 12, 1)), 
            'num_leaves': scope.int(hp.quniform('num_leaves', 90, 140, 5)), 
            'max_bin': scope.int(hp.quniform('max_bin', 140, 190, 2)),
            'scale_pos_weight': hp.uniform('scale_pos_weight', 1, 12),
            'feature_fraction': hp.loguniform('feature_fraction', np.log(0.6), np.log(0.8)), 
            'bagging_fraction': hp.loguniform('bagging_fraction', np.log(0.6), np.log(0.8)), 
            'bagging_freq': scope.int(hp.quniform('bagging_freq', 4, 8, 1)), 
            'boosting_type': 'gbdt', 
            'objective': 'binary',
            'metric': 'auc'
        }
        self._best_hyperparameters = None
        
        self._categorical_count = {}
        self._categorical_event = {}
        self._categorical_woe = {}
        self._time_map = {}
        for col_index in np.arange(info['time_starting_index'], info['numerical_starting_index']):
            self._time_map[col_index] = 0.0
        
    def fit(self, F, y, datainfo, timeinfo):
        print('\nEntering fit function')

        info = self._extract(datainfo, timeinfo)
        self._print_time_info(info)

        data = self._preprocess_time_and_numerical_data(F['numerical'], info)

        if info['no_of_categorical_features'] > 0:
            categorical_data = self._preprocess_categorical_data(F['CAT'], y)
            data = np.concatenate((data, categorical_data), axis=1)

        # if info['no_of_mvc_features'] > 0:    
        #     mvc_data = self._preprocess_mvc_data(F['MV'])
        #     data = np.concatenate((data, mvc_data), axis=1)
        
        print('data.shape: {}'.format(data.shape))
        print('y.shape: {}'.format(y.shape))

        self._training_data = data if len(self._training_data) == 0 else np.concatenate((self._training_data, data), axis=0)
        self._training_labels = y if len(self._training_labels) == 0 else np.concatenate((self._training_labels, y), axis=0)

        print('self._training_data.shape: {}'.format(self._training_data.shape))
        print('self._training_labels.shape: {}'.format(self._training_labels.shape))

        if self._too_much_training_data():
            print('\nRemoving excess data')
            self._remove_excess_training_data()
            print('self._training_data.shape: {}'.format(self._training_data.shape))
            print('self._training_labels.shape: {}\n'.format(self._training_labels.shape))

        if self._classifier is None or self._best_hyperparameters is None:
            tuner = HyperparametersTuner(self._classifier_class, self._fixed_hyperparameters, self._search_space)
            self._best_hyperparameters = tuner.get_best_hyperparameters(self._training_data, self._training_labels)

            print('self._best_hyperparameters: {}'.format(self._best_hyperparameters))

            self._classifier = self._classifier_class()
            self._classifier.set_params(**self._best_hyperparameters)
        
        self._classifier.fit(self._training_data, self._training_labels)

    def predict(self, F, datainfo, timeinfo):
        print('\nEntering predict function')

        info = self._extract(datainfo, timeinfo)
        self._print_time_info(info)

        data = self._preprocess_time_and_numerical_data(F['numerical'], info)

        if info['no_of_categorical_features'] > 0:
            categorical_data = self._preprocess_categorical_data(F['CAT'])
            data = np.concatenate((data, categorical_data), axis=1)

        # if info['no_of_mvc_features'] > 0:    
        #     mvc_data = self._preprocess_mvc_data(F['MV'])
        #     data = np.concatenate((data, mvc_data), axis=1)

        print('data.shape: {}'.format(data.shape))

        probabilities = self._classifier.predict_proba(data)[:,1]
        print('probabilities.shape: {}\n'.format(probabilities.shape))
        return probabilities
  
    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile) as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self

    def _preprocess_time_and_numerical_data(self, data, info):
        print('\nPreprocessing time and numerical data')
        data = np.nan_to_num(data)
        print('data.shape: {}'.format(data.shape))

        if info['no_of_time_features'] > 0:

            result = []
            for col_index in np.arange(info['time_starting_index'], info['numerical_starting_index']):
                date_col = data[:,col_index].astype(float)
                non_zero_indices = np.nonzero(date_col)[0]

                if len(non_zero_indices) != 0:
                    if self._time_map[col_index] == 0:
                        self._time_map[col_index] = np.min(date_col[non_zero_indices])
                    else:
                        self._time_map[col_index] = np.min([self._time_map[col_index], \
                                                            np.min(date_col[non_zero_indices])])

                transformed_date_col = data[:,col_index].astype(float) - self._time_map[col_index]
                result.append(transformed_date_col)
            
            for i in range(info['no_of_time_features']):
                for j in range(i+1, info['no_of_time_features']):
                    if len(np.nonzero(data[:,i])) > 0 and len(np.nonzero(data[:,j])) > 0:
                        result.append(data[:,i] - data[:,j])

                dates = pd.DatetimeIndex(data[:,i])
                dayofweek = dates.dayofweek.values
                dayofyear = dates.dayofyear.values
                month = dates.month.values
                weekofyear = dates.weekofyear.values
                day = dates.day.values
                hour = dates.hour.values
                minute = dates.minute.values
                year = dates.year.values

                result.append(dayofweek)
                result.append(dayofyear)
                result.append(month)
                result.append(weekofyear)
                result.append(year)
                result.append(day)
                result.append(hour)
                result.append(minute)

            result = np.array(result).T
            print('result.shape: {}'.format(result.shape))
            data = np.concatenate((result, data[:,info['numerical_starting_index']:]), axis=1)
        
        print('data.shape: {}'.format(data.shape))
        return data

    def _preprocess_categorical_data(self, data, labels=None):
        print('\nPreprocessing categorical data')
        data = data.fillna('nan')
        print('data.shape: {}'.format(data.values.shape))
        indices = data.dtypes.index

        result = []
        if labels is None: # predict
            for i in indices:
                d0 = pd.DataFrame({'X': data[i]})
                d1 = d0.join(self._categorical_woe[i], on='X')[['COUNT', 'WOE', 'IV']].values
                result = d1 if len(result) == 0 else np.concatenate((result, d1), axis=1)

                del d0
            
        else: # fit
            for i in indices:
                d0 = pd.DataFrame({'X': data[i], 'Y': labels.ravel()})
                d1 = d0.groupby('X',as_index=True)
                
                d2 = pd.DataFrame({},index=[])
                d2['COUNT'] = d1.count().Y
                d2['EVENT'] = d1.sum().Y

                self._categorical_count[i] = d2['COUNT'] if i in self._categorical_count else self._categorical_count[i] + d2['COUNT'] 
                self._categorical_event[i] = d2['EVENT'] if i in self._categorical_event else self._categorical_event[i] + d2['EVENT']

                d2['COUNT'] = self._categorical_count[i]
                d2['EVENT'] = self._categorical_event[i]
                d2['NONEVENT'] = d2.COUNT - d2.EVENT
                d2['DIST_EVENT'] = d2.EVENT/d2.sum().EVENT
                d2['DIST_NON_EVENT'] = d2.NONEVENT/d2.sum().NONEVENT
                d2['WOE'] = np.log(d2.DIST_EVENT/d2.DIST_NON_EVENT)
                d2['IV'] = (d2.DIST_EVENT - d2.DIST_NON_EVENT) * d2.WOE
                d2 = d2[['COUNT','WOE', 'IV']] 
                d2 = d2.replace([np.inf, -np.inf], 0)
                d2.IV = d2.IV.sum()

                self._categorical_woe[i] = d2
                d3 = d0.join(d2, on='X')[['COUNT', 'WOE', 'IV']].values
                result = d3 if len(result) == 0 else np.concatenate((result, d3), axis=1)
                
                del d0
                del d1

        result = np.array(result)
        print('result.shape: {}'.format(result.shape)) 
        return result
 
    def _preprocess_mvc_data(self, data):
        print('\nPreprocessing mvc data')
        data = data.fillna('nan')

    def _too_much_training_data(self):
        return len(self._training_data) > self._max_training_data

    def _remove_excess_training_data(self):
        total_training_data = len(self._training_data)
        remove_percentage = 1.0 - (float(self._max_training_data) / total_training_data)
        remove_samples = int(total_training_data * remove_percentage)
        skip = sorted(random.sample(range(total_training_data), total_training_data - remove_samples))
        
        self._training_data = self._training_data[skip,:]
        self._training_labels = self._training_labels[skip,:]

    def _extract(self, datainfo, timeinfo):
        time_budget = datainfo['time_budget']
        no_of_time_features = datainfo['loaded_feat_types'][0]
        no_of_numerical_features = datainfo['loaded_feat_types'][1]
        no_of_categorical_features = datainfo['loaded_feat_types'][2]
        no_of_mvc_features = datainfo['loaded_feat_types'][3]
        total_no_of_features = no_of_time_features + no_of_numerical_features + \
                            no_of_categorical_features + no_of_mvc_features

        time_starting_index = 0
        numerical_starting_index = no_of_time_features
        categorical_starting_index = numerical_starting_index + no_of_numerical_features
        mvc_starting_index = categorical_starting_index + no_of_categorical_features

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
            'time_starting_index': time_starting_index,
            'numerical_starting_index': numerical_starting_index,
            'categorical_starting_index': categorical_starting_index,
            'mvc_starting_index': mvc_starting_index,
            'overall_time_spent': overall_time_spent,
            'dataset_time_spent': dataset_time_spent
        }

    def _print_data_info(self, info):
        print('\nDataset budget: {0:f} seconds'.format(info['time_budget']))
        print('No. of time features: {0:d}'.format(info['no_of_time_features']))
        print('No. of numerical features: {0:d}'.format(info['no_of_numerical_features']))
        print('No. of categorical features: {0:d}'.format(info['no_of_categorical_features']))
        print('No. of mvc features: {0:d}\n'.format(info['no_of_mvc_features']))

    def _print_time_info(self, info):
        print('\nOverall time spent: {0:5.2f} seconds'.format(info['overall_time_spent']))
        print('Dataset time spent: {0:5.2f} seconds\n'.format(info['dataset_time_spent'])) 
