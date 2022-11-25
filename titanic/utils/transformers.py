import sys
import numpy as np

import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

sys.path.append('././')
from titanic.utils.custom_functions import get_title

class ReplaceQM(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.replace('?',np.nan)

class CleanData(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X[:,1] =  X[:,1].astype('float')
        X[:,1] = list(map(lambda x: round(x, 0),  X[:,1]))
        X[:,1] =  X[:,1].astype('int')

        X[:,4] =  X[:,4].astype('float')
        X[:,4] = list(map(lambda x: round(x, 0),  X[:,4]))

        return X

class AddColumns(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X['title'] = list(map(lambda x: get_title(x), X['name']))
        X['cabin'] = X['cabin'].astype('str')
        X['cabin_letter'] = list(map(lambda x: x[0], X['cabin']))
        X['cabin_letter'] = X['cabin_letter'].replace('n','U')
        return X