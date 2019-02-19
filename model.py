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
            'n_estimators': 500, 
            'max_depth': 7, 
            'num_leaves': 84, 
            'max_bin': 75,
            'feature_fraction': 0.6564234953698808, 
            'bagging_fraction': 0.38713947428926554, 
            'bagging_freq': 4, 
            'boosting_type': 'gbdt', 
            'objective': 'binary', 
            'metric': 'auc' 
        }
        self._search_space = {
            'n_estimators': scope.int(hp.quniform('n_estimators', 400, 600, 25)), 
            'max_depth': scope.int(hp.quniform('max_depth', 5, 7, 1)), 
            'num_leaves': scope.int(hp.quniform('num_leaves', 70, 100, 2)), 
            'max_bin': scope.int(hp.quniform('max_bin', 70, 100, 3)),
            'feature_fraction': hp.loguniform('feature_fraction', np.log(0.1), np.log(0.9)), 
            'bagging_fraction': hp.loguniform('bagging_fraction', np.log(0.1), np.log(0.9)), 
            'bagging_freq': scope.int(hp.quniform('bagging_freq', 2, 5, 1)), 
            'boosting_type': 'gbdt', 
            'objective': 'binary',
            'metric': 'auc'
        }
        self._best_hyperparameters = None
        
    def fit(self, F, y, datainfo, timeinfo):
        print('\nEntering fit function')

        info = self._extract(datainfo, timeinfo)
        self._print_time_info(info)
        
        data = self._fill_nan(F, info)
        print('data.shape: {}'.format(data.shape))
        print('y.shape: {}'.format(y.shape))

        if info['no_of_categorical_features'] > 0:
            print(F['CAT'].dtypes.index)
            cat = pd.DataFrame({'X': F['CAT']['6'], 'Y': y.ravel() })
            print(cat)
            d3 = pd.DataFrame({},index=[])
            d3["COUNT"] = cat.count().Y
            d3["EVENT"] = cat.sum().y
            d3["NONEVENT"] = count - event
            d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
            d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
            d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
            print('\nDataframe d3')
            print(d3)

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

        data = self._fill_nan(F, info)
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

    def _fill_nan(self, F, info):
        
        # Convert time and numerical nan
        data = F['numerical']
        data = np.nan_to_num(data)

        # Convert categorical nan
        # if info['no_of_categorical_features'] > 0:
            # categorical_data = F['CAT'].fillna('nan').values
            # data = np.concatenate((data, categorical_data), axis=1)
            # del categorical_data

        # Convert mvc nan
        # if info['no_of_mvc_features'] > 0:
        #     mvc_data = F['MV'].fillna('nan').values
        #     data = np.concatenate((data, mvc_data), axis=1)
        #     del mvc_data
        
        return data