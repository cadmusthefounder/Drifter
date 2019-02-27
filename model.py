import pickle
import os
from os.path import isfile

# cmd1 = 'apt-get -y update'
# cmd2 = 'apt-get -y install libdpkg-perl'
# os.system(cmd1)
# os.system(cmd2)

from architectures import ADWIN_VFDT

ARCHITECTURE = ADWIN_VFDT.NAME

architecture_mapping = {
    ADWIN_VFDT.NAME: ADWIN_VFDT
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