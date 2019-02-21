from math import pow
from random import random
import numpy as np

class BiasedReservoirSampler:

    def __init__(self, capacity, bias_rate):
        self._capacity = capacity
        self._bias_rate = bias_rate
        self._p_in = self._capacity * self._bias_rate
        self._q = pow(self._capacity, -1)

    def sample(self, current_reservoir_data, current_reservoir_label, incoming_data, incoming_label):
        print('\nsample')
        print('before sampling current_reservoir_data.shape: {}'.format(current_reservoir_data.shape))
        print('before sampling current_reservoir_label.shape: {}'.format(current_reservoir_label.shape))
        print('incoming_data.shape: {}'.format(incoming_data.shape))
        print('incoming_label.shape: {}'.format(incoming_label.shape))

        indices = np.random.randint(self._capacity, size=len(incoming_data))

        for i in range(len(incoming_data)):
            if len(current_reservoir_data) < self._capacity or self._triggered(self._p_in):
                if indices[i] < len(current_reservoir_data):
                    current_reservoir_data[j] = incoming_data[i]
                    current_reservoir_label[j] = incoming_label[i]
                else:
                    current_reservoir_data = [incoming_data[i]] if len(current_reservoir_data) == 0 else \
                                                np.append(current_reservoir_data, [incoming_data[i]], axis=0)
                    current_reservoir_label = [incoming_label[i]] if len(current_reservoir_label) == 0 else \
                                                np.append(current_reservoir_label, [incoming_label[i]], axis=0) 

        print('after sampling current_reservoir_data.shape: {}'.format(current_reservoir_data.shape))
        print('after sampling current_reservoir_label.shape: {}\n'.format(current_reservoir_label.shape))
        return current_reservoir_data, current_reservoir_label

    def _triggered(self, probability):
        return random() <= probability