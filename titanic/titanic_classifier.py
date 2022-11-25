import sys
import numpy as np

import joblib
from pathlib import Path
import pandas as pd

sys.path.append('../')
from titanic.utils.pipelines import TitanicClassifierPipeline
from titanic.utils.models import Passanger

class TitanicClassifier():
    def __init__(self):

        self.model_path = Path(__file__).parent.joinpath(Path('estimators\\titanic_classifier.pkl'))
        self.survived = {
            0: 'dead',
            1: 'survived',
            }
        
        # This attribute will hold the raw training data of the model 
        self.TITANIC_TRAIN_DATA = None
        self.X = None
        self.y = None
        self.estimator = None

    def read_data(self):
        self.TITANIC_TRAIN_DATA = pd.read_csv(r'https://www.openml.org/data/get_csv/16826755/phpMYEkMl')
        self.X = self.TITANIC_TRAIN_DATA.drop(['survived', 'ticket', 'boat', 'body', 'home.dest'], axis = 1)
        self.y = self.TITANIC_TRAIN_DATA['survived'].copy()
    
    def train(self):
        self.read_data()
        self.estimator = TitanicClassifierPipeline().titanic_pipeline.fit(self.X, self.y)
        joblib.dump(self.estimator, self.model_path)
    
    def load_model(self):
        self.estimator = joblib.load(self.model_path)
    
    def sink_titanic(self, passanger: Passanger):
        X = [
            passanger.pclass,
            passanger.name, 
            str(passanger.sex), 
            passanger.age, 
            passanger.sibsp,
            passanger.parch,
            passanger.fare, 
            passanger.cabin, 
            str(passanger.embarked)
            ]
        pred = self.estimator.predict_proba(
            pd.DataFrame(
                [X],
                columns = [
                    'pclass',
                    'name',
                    'sex',
                    'age',
                    'sibsp',
                    'parch',
                    'fare',
                    'cabin',
                    'embarked'
                    ]
             ))
        print("The Titanic has sunk, getting the survival probability...")
        return {'survived?': self.survived[np.argmax(pred)],
                'probability': str(int(round(max(pred[0]), 2)*100))+"%"}





if __name__ == '__main__':
    a = TitanicClassifier()
    a.train()
    a.load_model()