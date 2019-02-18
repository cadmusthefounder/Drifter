import pickle
import time
from os.path import isfile

from utils import pip_install

pip_install('hyperopt')
pip_install('lightgbm')

from lightgbm import LGBMClassifier

class Model:

    def __init__(self, datainfo, timeinfo):
        info = self._extract(datainfo, timeinfo)
        
        self._max_training_data = 200000
        self._training_data = np.array([])
        self._training_labels = np.array([])
        self._clf = LGBMClassifier()
        self._hyperparameters = {
            'learning_rate': 0.01, 
            'n_estimators': 400, 
            'max_depth': 12, 
            'num_leaves': 50, 
            'max_bin': 150,
            'feature_fraction': 0.6, 
            'bagging_fraction': 0.6, 
            'bagging_freq': 2, 
            'boosting_type': 'gbdt', 
            'objective': 'binary', 
            'metric': 'auc' 
        }
        
    def fit(self, F, y, datainfo, timeinfo):
        print('Entering fit function')

        info = self._extract(datainfo, timeinfo)
        data = self._fill_nan(F, info)

        print('data.shape: {}'.format(data.shape))
        print('y.shape: {}'.format(y.shape))

        self._training_data = np.concatenate((self._training_data, data), axis=0)
        self._training_labels = np.concatenate((self._training_labels, y), axis=0)

        print('self._training_data.shape: {}'.format(self._training_data.shape))
        print('self._training_labels.shape: {}'.format(self._training_labels.shape))

        if self._too_much_training_data():
            print('Removing excess data')
            self._remove_excess_training_data()
            print('self._training_data.shape: {}'.format(self._training_data.shape))
            print('self._training_labels.shape: {}'.format(self._training_labels.shape))

        self._clf.set_params(**self._hyperparameters)
        self._clf.fit(self._training_data, self._training_labels)

    def predict(self, F, datainfo, timeinfo):
        print('Entering predict function')

        info = self._extract(datainfo, timeinfo)
        data = self._fill_nan(F, info)
        print('data.shape: {}'.format(data.shape))

        probabilities = self._clf.predict_proba(data)
        print('probabilities.shape: {}'.format(probabilities.shape))
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

    def _fill_nan(self, F, info):
        
        # Convert time and numerical nan
        data = F['numerical']
        data = np.nan_to_num(data)

        # Convert categorical nan
        # if info['no_of_categorical_features'] > 0:
        #     categorical_data = F['CAT'].fillna('nan').values
        #     data = np.concatenate((data, categorical_data), axis=1)
        #     del categorical_data

        # Convert mvc nan
        # if info['no_of_mvc_features'] > 0:
        #     mvc_data = F['MV'].fillna('nan').values
        #     data = np.concatenate((data, mvc_data), axis=1)
        #     del mvc_data
        
        return data