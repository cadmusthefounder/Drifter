import numpy as np
import pandas as pd

class CountWoeEncoder:

    def __init__(self):
        self._count_woe_map = {}

    def encode(self, incoming_data, incoming_labels=None):
        print('\nencode')
        print('incoming_data.shape: {}'.format(incoming_data.shape))
        print('incoming_labels.shape: {}'.format(incoming_labels.shape))

        no_of_rows, no_of_cols = incoming_data.shape
        result = np.array([])
        if incoming_labels is None and self._count_woe_map:
            for i in range(no_of_cols):
                d0 = pd.DataFrame({'X': incoming_data[:,i]})
                d1 = d0.join(self._count_woe_map[i], on='X')[['COUNT', 'WOE']].values
                result = d1 if len(result) == 0 else np.concatenate((result, d1), axis=1)
                del d0
        else:
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
