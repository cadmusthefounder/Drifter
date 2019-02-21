from math import pow
from random import random
import numpy as np

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
        self._current_reservoir_data = np.empty([self._capacity, info['total_no_of_features']])
        self._current_reservoir_label = np.empty(self._capacity)
        
    def sample(self, incoming_data, incoming_label):
        print('\nsample')
        print('before sampling self._current_reservoir_data.shape: {}'.format(self._current_reservoir_data.shape))
        print('before sampling self._current_reservoir_label.shape: {}'.format(self._current_reservoir_label.shape))
        print('incoming_data.shape: {}'.format(incoming_data.shape))
        print('incoming_label.shape: {}'.format(incoming_label.shape))
        
        for i in range(len(incoming_data)):
            if self._current_capacity < self._capacity or self._triggered():
                if self._current_index >= len(self._indices):
                    self._current_index = 0
                    self._indices = self._generate_indices()

                if self._indices[self._current_index] < self._current_capacity:
                    self._current_reservoir_data[self._indices[self._current_index]] = incoming_data[i]
                    self._current_reservoir_label[self._indices[self._current_index]] = incoming_label[i]
                else:
                    self._current_reservoir_data[self._current_capacity] = incoming_data[i]
                    self._current_reservoir_label[self._current_capacity] = incoming_label[i]
                    self._current_capacity += 1
                
                self._current_index += 1

        print('after sampling self._current_reservoir_data.shape: {}'.format(self._current_reservoir_data.shape))
        print('after sampling self._current_reservoir_label.shape: {}\n'.format(self._current_reservoir_label.shape))
        return current_reservoir_data, current_reservoir_label

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