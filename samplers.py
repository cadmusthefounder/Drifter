from math import pow
from random import random
import numpy as np

from utils import pip_install
pip_install('imbalanced-learn')

from imblearn.over_sampling import SMOTE, SMOTENC

class BiasedReservoirSampler:

    def __init__(self, capacity, bias_rate, info):
        self._capacity = capacity
        self._bias_rate = bias_rate
        self._size = 2000000
        self._p_in = self._capacity * self._bias_rate
        self._p_in_index = 0
        self._p_in_array = self._generate_p_in_array()
        
        self._current_index = 0
        self._indices = self._generate_indices()
        
        self._current_capacity = 0
        self._current_reservoir_data = np.empty([self._capacity, info['total_no_of_features']], dtype=object)
        self._current_reservoir_label = np.empty(self._capacity)
        
    def sample(self, incoming_data, incoming_labels):
        print('\nsample')
        print('incoming_data.shape: {}'.format(incoming_data.shape))
        print('incoming_labels.shape: {}'.format(incoming_labels.shape))
        
        for i in range(len(incoming_data)):
            if self._current_capacity < self._capacity or self._triggered():
                if self._current_index >= len(self._indices):
                    self._current_index = 0
                    self._indices = self._generate_indices()

                if self._indices[self._current_index] < self._current_capacity:
                    self._current_reservoir_data[self._indices[self._current_index]] = incoming_data[i]
                    self._current_reservoir_label[self._indices[self._current_index]] = incoming_labels[i]
                else:
                    self._current_reservoir_data[self._current_capacity] = incoming_data[i]
                    self._current_reservoir_label[self._current_capacity] = incoming_labels[i]
                    self._current_capacity += 1
                
                self._current_index += 1

        actual_reservoir_data = self._current_reservoir_data
        actual_reservoir_labels = self._current_reservoir_label
        if self._current_capacity < self._capacity:
            actual_reservoir_data = actual_reservoir_data[:self._current_capacity,:]
            actual_reservoir_labels = actual_reservoir_labels[:self._current_capacity]


        print('actual_reservoir_data.shape: {}'.format(actual_reservoir_data.shape))
        print('actual_reservoir_labels.shape : {}\n'.format(actual_reservoir_labels))
        return actual_reservoir_data, actual_reservoir_labels

    def _triggered(self):
        if self._p_in_index >= len(self._p_in_array):
            self._p_in_index = 0
            self._p_in_array = self._generate_p_in_array()

        triggered = self._p_in_array[self._p_in_index] <= self._p_in
        self._p_in_index += 1
        return triggered

    def _generate_indices(self):
        return np.random.randint(self._capacity, size=self._size)

    def _generate_p_in_array(self):
        return np.random.random_sample(size=self._size)

class SMOTENCSampler:

    def __init__(self, info, random_state=42):
        self._random_state = random_state
        if info['no_of_categorical_features'] > 0:
            cat_features = list(range(info['categorical_data_starting_index'], info['total_no_of_features']))
            self._smotenc_sampler = SMOTENC(cat_features, random_state=self._random_state)
        elif info['no_of_mvc_features'] > 0:
            cat_features = list(range(info['mvc_starting_index'], info['total_no_of_features']))
            self._smotenc_sampler = SMOTENC(cat_features, random_state=self._random_state)
        else:
            self._smotenc_sampler = SMOTE(random_state=self._random_state)

    def sample(self, incoming_data, incoming_labels):
        print('\nsample')
        sampled_data, sampled_labels = self._smotenc_sampler.fit_resample(incoming_data, incoming_labels)

        print('sampled_data.shape: {}'.format(sampled_data.shape))
        print('sampeld_labels.shape: {}\n'.format(sampled_labels.shape))
        return sampled_data, sampled_labels