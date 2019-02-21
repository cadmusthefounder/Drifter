import pickle
from os.path import isfile

import numpy as np

from architectures import SMOTENC_BiasedReservoirSampler_LightGBM

ARCHITECTURE = SMOTENC_BiasedReservoirSampler_LightGBM.NAME

architecture_mapping = {
    SMOTENC_BiasedReservoirSampler_LightGBM.NAME: SMOTENC_BiasedReservoirSampler_LightGBM
}

class Model:

    def __init__(self, datainfo, timeinfo):
        self._architecture = architecture_mapping[ARCHITECTURE](datainfo, timeinfo)
        
    def fit(self, F, y, datainfo, timeinfo):
        print(F['numerical'][:,1])
        print(np.nan_to_num(F['numerical'][:,1]))

        X = np.nan_to_num(F['numerical'][:,:2])
        print(X.shape)
        print(X[:,1])

        
        first = X[:,0]
        print(np.where(first == 0)[0].shape)

        second = X[:,1]
        print(np.where(second == 0)[0].shape)


        self._architecture.fit(F, y, datainfo, timeinfo)

    def predict(self, F, datainfo, timeinfo):
        return self._architecture.predict(F, datainfo, timeinfo)

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile) as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self