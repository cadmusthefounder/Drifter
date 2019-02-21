from math import pow
from random import random
import numpy as np

class BiasedReservoirSampler:

    def __init__(self, capacity, bias_rate):
        self._capacity = capacity
        self._bias_rate = bias_rate
        self._p_in = self._capacity * self._bias_rate
        self._size = 2000000
        self._indices = self._generate_indices(self._capacity, self._size)
        self._current_index = 0
        
    def sample(self, current_reservoir_data, current_reservoir_label, incoming_data, incoming_label):
        print('\nsample')
        print('before sampling current_reservoir_data.shape: {}'.format(current_reservoir_data.shape))
        print('before sampling current_reservoir_label.shape: {}'.format(current_reservoir_label.shape))
        print('incoming_data.shape: {}'.format(incoming_data.shape))
        print('incoming_label.shape: {}'.format(incoming_label.shape))

        

        for i in range(len(incoming_data)):
            if len(current_reservoir_data) < self._capacity or self._triggered(self._p_in):
                if self._current_index >= len(self._indices):
                    self._current_index = 0
                    self._indices = self._generate_indices(self._capacity, self._size)

                if self._indices[self._current_index] < len(current_reservoir_data):
                    current_reservoir_data[self._indices[self._current_index]] = incoming_data[i]
                    current_reservoir_label[self._indices[self._current_index]] = incoming_label[i]
                else:
                    current_reservoir_data = [incoming_data[i]] if len(current_reservoir_data) == 0 else \
                                                np.append(current_reservoir_data, [incoming_data[i]], axis=0)
                    current_reservoir_label = [incoming_label[i]] if len(current_reservoir_label) == 0 else \
                                                np.append(current_reservoir_label, [incoming_label[i]], axis=0) 
                
                self._current_index += 1

        print('after sampling current_reservoir_data.shape: {}'.format(current_reservoir_data.shape))
        print('after sampling current_reservoir_label.shape: {}\n'.format(current_reservoir_label.shape))
        return current_reservoir_data, current_reservoir_label

    def _triggered(self, probability):
        return random() <= probability

    def _generate_indices(self, capacity, size):
        return np.random.randint(capacity, size=size)