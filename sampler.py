from math import pow
from random import random, randint
import numpy as np

class BiasedReservoirSampler:
    NAME = 'BiasedReservoirSampler'

    def __init__(self, capacity, bias_rate):
        self._capacity = capacity
        self._bias_rate = bias_rate
        self._p_in = 1
        self._q = pow(self._capacity, -1)

    def sample(self, current_reservoir_data, current_reservoir_label, incoming_data, incoming_label):
        for i in range(len(incoming_data)):
            if len(current_reservoir_data) <= self._capacity:
                fraction_filled = float(len(current_reservoir_data)) / float(self._capacity)
                if self._triggered(self._p_in):
                    if self._triggered(fraction_filled) and len(current_reservoir_data) > 0:
                        j = randint(0, len(current_reservoir_data))
                        current_reservoir_data[j] = incoming_data[i]
                        current_reservoir_label[j] = incoming_label[i]
                    else:
                        current_reservoir_data = [incoming_data[i]] if len(current_reservoir_data) == 0 else \
                                                    np.append(current_reservoir_data, [incoming_data[i]], axis=0)
                        current_reservoir_label = [incoming_label[i]] if len(current_reservoir_label) == 0 else \
                                                    np.append(current_reservoir_label, [incoming_label[i]], axis=0) 
            elif self._p_in > self._capacity * self._bias_rate:
                self._p_in *= (1 - self._q)
                no_of_points_to_delete = int(self._q * self._capacity)

                for j in range(no_of_points_to_delete)
                    k = randint(0, len(current_reservoir_data))
                    current_reservoir_data = np.delete(current_reservoir_data, k, 0)
                    current_reservoir_label = np.delete(current_reservoir_label, k)
            elif self._p_in <= self._capacity * self._bias_rate:
                self._p = self._capacity * self._bias_rate

        return current_reservoir_data, current_reservoir_label

    def _triggered(self, probability):
        return random() <= probability