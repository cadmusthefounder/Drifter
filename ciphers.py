import numpy as np
import pandas as pd

from utils import pip_install

pip_install('category_encoders')
from category_encoders.binary import BinaryEncoder

class CountWoeCipher:

    def __init__(self):
        self._count_woe_map = {}

    def encode(self, incoming_data, incoming_labels=None):
        print('\nencode')
        print('incoming_data.shape: {}'.format(incoming_data.shape))

        no_of_rows, no_of_cols = incoming_data.shape
        result = np.array([])
        if incoming_labels is None and self._count_woe_map: # predict
            for i in range(no_of_cols):
                d0 = pd.DataFrame({'X': incoming_data[:,i]})
                d1 = d0.join(self._count_woe_map[i], on='X')[['COUNT', 'WOE']].values
                result = d1 if len(result) == 0 else np.concatenate((result, d1), axis=1)
                del d0
        else: # fit
            print('incoming_labels.shape: {}'.format(incoming_labels.shape))
            for i in range(no_of_cols):
                d0 = pd.DataFrame({'X': incoming_data[:,i], 'Y': incoming_labels})
                d1 = d0.groupby('X',as_index=True)
                d2 = pd.DataFrame({},index=[])
                d2['COUNT'] = d1.count().Y
                d2['EVENT'] = d1.sum().Y
                d2['NONEVENT'] = d2.COUNT - d2.EVENT
                d2['DIST_EVENT'] = d2.EVENT/d2.sum().EVENT
                d2['DIST_NON_EVENT'] = d2.NONEVENT/d2.sum().NONEVENT
                d2['WOE'] = np.log(d2.DIST_EVENT/d2.DIST_NON_EVENT)
                # d2['IV'] = (d2.DIST_EVENT - d2.DIST_NON_EVENT) * d2.WOE
                d2 = d2[['COUNT', 'WOE']] 
                d2 = d2.replace([np.inf, -np.inf], 0)
                # d2.IV = d2.IV.sum()

                self._count_woe_map[i] = d2
                d3 = d0.join(d2, on='X')[['COUNT', 'WOE']].values
                result = d3 if len(result) == 0 else np.concatenate((result, d3), axis=1)   
                del d0
                del d1
        print('result.shape: {}\n'.format(result.shape)) 
        return result

class BinaryCipher:

    def __init__(self):
        self._binary_encoder = BinaryEncoder(return_df=False, handle_unknown='ignore')

    def encode(self, incoming_data, incoming_labels=None):
        print('\nencode')
        print('incoming_data.shape: {}'.format(incoming_data.shape))
        result = np.array([])
        if incoming_labels is None: # predict
            result = self._binary_encoder.transform(incoming_data)
        else: #fit
            print('incoming_labels.shape: {}'.format(incoming_labels.shape))
            result = self._binary_encoder.fit_transform(incoming_data, incoming_labels)

        print('result.shape: {}\n'.format(result.shape)) 
        return result

# def hash_encoding(categorical_or_mvc_data, labels=None, encoder=None):
#     print('\nhash')
    # result = np.array([])
    # if labels is None and encoder is not None: # predict
    #     result = encoder.transform(categorical_or_mvc_data)
    # else: #fit
    #     encoder = HashingEncoder(cols=list(range(categorical_or_mvc_data.shape[1])), n_components=10)
    #     result = encoder.fit_transform(categorical_or_mvc_data, labels)

    # print('result.shape: {}\n'.format(result.shape)) 
    # return result