from math import pow
from utils import *

pip_install('lightgbm')
pip_install('hyperopt')

import numpy as np

from lightgbm import LGBMClassifier
from hyperopt import hp
from hyperopt.pyll.base import scope
from hyperparameters_tuner import HyperparametersTuner
from sklearn.metrics import roc_auc_score

from samplers import BiasedReservoirSampler, BorderlineSMOTESampler
from ciphers import CountWoeCipher, BinaryCipher

class SMOTENC_BiasedReservoirSampler_LightGBM:
    NAME = 'SMOTENC_BiasedReservoirSampler_LightGBM'
    
    def __init__(self, datainfo, timeinfo):
        info = extract(datainfo, timeinfo)
        print_data_info(info)
        print_time_info(info)
        
        self._capacity = 350000
        self._bias_rate = pow(10, -6)
        self._biased_reservoir_sampler = BiasedReservoirSampler(self._capacity, self._bias_rate, info)
        self._borderline_smote_sampler = BorderlineSMOTESampler()
        
        self._dataset_budget_threshold = 0.8
        self._cat_encoder = CountWoeCipher()
        self._mvc_encoder = CountWoeCipher()
        # self._cat_encoder = BinaryCipher()
        # self._mvc_encoder = BinaryCipher()
        
        self._classifier = None
        self._classifier_class = LGBMClassifier
        self._fixed_hyperparameters = {
            'learning_rate': 0.01, 
            'n_estimators': 530, 
            'max_depth': 11, 
            'num_leaves': 110, 
            'max_bin': 146,
            'feature_fraction': 0.6290429279076984, 
            'bagging_fraction': 0.7890879497539331, 
            'bagging_freq': 6, 
            'boosting_type': 'gbdt', 
            'objective': 'binary', 
            'metric': 'auc'
        }
        self._search_space = {
            'n_estimators': scope.int(hp.quniform('n_estimators', 500, 700, 10)), 
            'max_depth': scope.int(hp.quniform('max_depth', 7, 14, 1)), 
            'num_leaves': scope.int(hp.quniform('num_leaves', 120, 160, 5)), 
            'max_bin': scope.int(hp.quniform('max_bin', 140, 190, 8)),
            'feature_fraction': hp.loguniform('feature_fraction', np.log(0.6), np.log(0.9)), 
            'bagging_fraction': hp.loguniform('bagging_fraction', np.log(0.6), np.log(0.9)), 
            'bagging_freq': scope.int(hp.quniform('bagging_freq', 4, 10, 1)), 
            'boosting_type': 'gbdt', 
            'objective': 'binary',
            'metric': 'auc'
        }
        self._best_hyperparameters = None

    def fit(self, F, y, datainfo, timeinfo):
        print('\nfit')

        info = extract(datainfo, timeinfo)
        print_time_info(info)

        data = get_data(F, info)
        y = y.ravel()

        print('data.shape: {}'.format(data.shape))
        print('y.shape: {}'.format(y.shape))
        bincount = np.bincount(y.astype(int))
        print('Number of 0 label: {}'.format(bincount[0]))
        print('Number of 1 label: {}'.format(bincount[1]))
        
        if has_sufficient_time(self._dataset_budget_threshold, info) or self._classifier is None:

            transformed_data = np.array([])
            time_data, numerical_data, categorical_data, mvc_data = split_data_by_type(data, info)
            if len(time_data) > 0:
                transformed_data = subtract_min_time(time_data)
                transformed_data = np.concatenate((transformed_data, difference_between_time_columns(time_data)), axis=1)
            if len(numerical_data) > 0:
                transformed_data = numerical_data if len(transformed_data) == 0 else \
                                    np.concatenate((transformed_data, numerical_data), axis=1)
            if len(categorical_data) > 0:
                encoded_categorical_data = self._cat_encoder.encode(categorical_data, incoming_labels=y)
                transformed_data = np.concatenate((transformed_data, encoded_categorical_data), axis=1)
            if len(mvc_data) > 0:
                encoded_mvc_data = self._mvc_encoder.encode(mvc_data, incoming_labels=y)
                transformed_data = np.concatenate((transformed_data, encoded_mvc_data), axis=1)

            print('transformed_data.shape: {}'.format(transformed_data.shape))
            sampled_training_data, sampled_training_labels = self._borderline_smote_sampler.sample(transformed_data, y)
            sampled_training_data, sampled_training_labels = self._biased_reservoir_sampler.sample(sampled_training_data, sampled_training_labels)

            if self._best_hyperparameters is None:
                tuner = HyperparametersTuner(self._classifier_class, self._fixed_hyperparameters, self._search_space)
                self._best_hyperparameters = tuner.get_best_hyperparameters(sampled_training_data, sampled_training_labels)

                print('self._best_hyperparameters: {}\n'.format(self._best_hyperparameters))

                self._classifier = self._classifier_class()
                self._classifier.set_params(**self._best_hyperparameters)
        
        self._classifier.fit(transformed_data, sampled_training_labels)

    def predict(self, F, datainfo, timeinfo):
        print('\npredict')

        info = extract(datainfo, timeinfo)
        print_time_info(info)

        data = get_data(F, info)
        print('data.shape: {}'.format(data.shape))

        transformed_data = np.array([])
        time_data, numerical_data, categorical_data, mvc_data = split_data_by_type(data, info)
        if len(time_data) > 0:
            transformed_data = subtract_min_time(time_data)
            transformed_data = np.concatenate((transformed_data, difference_between_time_columns(time_data)), axis=1)
        if len(numerical_data) > 0:
            transformed_data = numerical_data if len(transformed_data) == 0 else \
                                np.concatenate((transformed_data, numerical_data), axis=1)
        if len(categorical_data) > 0:
            encoded_categorical_data = self._cat_encoder.encode(categorical_data)
            transformed_data = np.concatenate((transformed_data, encoded_categorical_data), axis=1)
        if len(mvc_data) > 0:
            encoded_mvc_data = self._mvc_encoder.encode(mvc_data)
            transformed_data = np.concatenate((transformed_data, encoded_mvc_data), axis=1)

        print('transformed_data.shape: {}'.format(transformed_data.shape))
        probabilities = self._classifier.predict_proba(transformed_data)[:,1]
        print('probabilities.shape: {}\n'.format(probabilities.shape))
        return probabilities