import sys

import joblib
from pathlib import Path
import pandas as pd

sys.path.append('./')
from titanic.utils.pipelines import TitanicClassifierPipeline

class TitanicClassifier():
    def __init__(self):

        self.model_path = Path(__file__).parent.joinpath(Path('estimators\\titanic_classifier.pkl'))
        
        # This attribute will hold the raw training data of the model 
        self.TITANIC_TRAIN_DATA = None
        self.X = None
        self.y = None
        self.estimator = None

    def read_data(self):
        self.TITANIC_TRAIN_DATA = pd.read_csv(r'https://www.openml.org/data/get_csv/16826755/phpMYEkMl')
        self.X = self.TITANIC_TRAIN_DATA.drop(['survived'], axis = 1)
        self.y = self.TITANIC_TRAIN_DATA['survived'].copy()
    
    def train(self):
        self.read_data()
        self.estimator = TitanicClassifierPipeline().titanic_pipeline.fit(self.X, self.y)
        joblib.dump(self.estimator, self.model_path)
    
    def load_model(self):
        self.estimator = joblib.load(self.model_path)





if __name__ == '__main__':
    a = TitanicClassifier()
    a.train()
    a.load_model()
    print(a.estimator)