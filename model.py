import pickle
from os.path import isfile

from architectures import BiasedReservoirSampler

ARCHITECTURE = BiasedReservoirSampler.NAME

architecture_mapping = {
    BiasedReservoirSampler.NAME: BiasedReservoirSampler
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