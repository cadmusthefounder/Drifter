from utils import *
pip_install('lightgbm')
pip_install('hyperopt')
# pip_install('scikit-multiflow')

import numpy as np
from math import pow
from lightgbm import LGBMClassifier
from hyperopt import hp
from hyperopt.pyll.base import scope
from hyperparameters_tuner import HyperparametersTuner
# from skmultiflow.drift_detection.adwin import ADWIN
from ciphers import HashCipher
from classifiers import Vfdt

class ADWIN_VFDT:
    NAME = 'ADWIN_VFDT'
    
    def __init__(self, datainfo, timeinfo):
        info = extract(datainfo, timeinfo)
        print_data_info(info)
        print_time_info(info)

        # self._adwin = ADWIN()
        self._dataset_budget_threshold = 0.8
        self._cat_encoder = HashCipher(info['no_of_categorical_features'], 3)
        self._mvc_encoder = HashCipher(info['no_of_mvc_features'], 6)
        
        self._classifier = None
        self._classifier_class = Vfdt
        self._fixed_hyperparameters = {}
        self._search_space = {}
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

        transformed_data = np.array([])
        time_data, numerical_data, categorical_data, mvc_data = split_data_by_type(data, info)
        if len(time_data) > 0:
            transformed_data = subtract_min_time(time_data)
            transformed_data = np.concatenate((transformed_data, difference_between_time_columns(time_data)), axis=1)
        if len(numerical_data) > 0:
            transformed_data = numerical_data if len(transformed_data) == 0 else \
                                np.concatenate((transformed_data, numerical_data), axis=1)
        if len(categorical_data) > 0:
            categorical_data = self._cat_encoder.encode(categorical_data)
            transformed_data = np.concatenate((transformed_data, categorical_data), axis=1)
        if len(mvc_data) > 0: 
            mvc_data = self._mvc_encoder.encode(mvc_data)
            transformed_data = np.concatenate((transformed_data, mvc_data), axis=1)

        print('transformed_data.shape: {}'.format(transformed_data.shape))
        split = 0
        if self._classifier is None:
            split = int(len(transformed_data) / 10)
            self._classifier = self._classifier_class(list(range(transformed_data.shape[1])))
            for i in range(split):
                current_data, current_label = transformed_data[i], y[i]
                self._classifier.update(current_data, current_label)
        for i in range(split, len(transformed_data)):
            current_data, current_label = transformed_data[i], y[i]
            prediction = self._classifier.predict([current_data])
            # print('prediction is: {} label is {}'.format(prediction, current_label))
            self._classifier.update(current_data, current_label)

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
        # if len(categorical_data) > 0:
        #     transformed_data = np.concatenate((transformed_data, categorical_data), axis=1)
        # if len(mvc_data) > 0:
        #     transformed_data = np.concatenate((transformed_data, mvc_data), axis=1)

        print('transformed_data.shape: {}'.format(transformed_data.shape))
        predictions = np.array(self._classifier.predict(transformed_data))
        print('predictions.shape: {}\n'.format(predictions.shape))
        return predictions
