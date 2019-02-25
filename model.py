import pickle
import os
from os.path import isfile

cmd1 = 'y | apt-get update'
cmd2 = 'y | apt-get install libdpkg-perl'

os.system(cmd1)
os.system(cmd2)

from architectures import SMOTENC_BiasedReservoirSampler_LightGBM

ARCHITECTURE = SMOTENC_BiasedReservoirSampler_LightGBM.NAME

architecture_mapping = {
    SMOTENC_BiasedReservoirSampler_LightGBM.NAME: SMOTENC_BiasedReservoirSampler_LightGBM
}

class Model:

    def __init__(self, datainfo, timeinfo):
        self._architecture = architecture_mapping[ARCHITECTURE](datainfo, timeinfo)
        
    def fit(self, F, y, datainfo, timeinfo):
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