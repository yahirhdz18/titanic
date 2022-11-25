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
            0: 'No',
            1: 'Yes',
            }
        
        # This attribute will hold the raw training data of the model 
        self.TITANIC_TRAIN_DATA = None
        self.X = None
        self.y = None
        self.estimator = None

    def read_data(self):
        r""" Loads the data of the Titanic.

        Loads the titanic data, then splits it into X and y,
        also dropping the columns in the csv that are not used during the model
        training.
        """
        self.TITANIC_TRAIN_DATA = pd.read_csv(r'https://www.openml.org/data/get_csv/16826755/phpMYEkMl')
        self.X = self.TITANIC_TRAIN_DATA.drop(['survived', 'ticket', 'boat', 'body', 'home.dest'], axis = 1)
        self.y = self.TITANIC_TRAIN_DATA['survived'].copy()
    
    def train(self):
        r""" Trains the titanic model.

        Loads the titanic data, then fits the prediction pipeline
        and finally dumps the model into the estimators folder.
        """
        self.read_data()
        self.estimator = TitanicClassifierPipeline().titanic_pipeline.fit(self.X, self.y)
        joblib.dump(self.estimator, self.model_path)
    
    def load_model(self):
        r""" Loads the trained model from a .pkl file.
        """
        self.estimator = joblib.load(self.model_path)
    
    def sink_titanic(self, passanger: Passanger)->dict:
        r""" Predicts the probability that a specific passanger will
        survive after the sinking of the Titanic.

        Adjusts the values of the Passanger instance into a list,
        and then converts it to a pd.DataFrame that matches the training
        data of the model and predicts the probability of surviving or
        dying.
        Returns a dictionary with the survival prediction and the probability.

        Parameters
        ----------
        passanger : Passanger
            The information of the Titanic Passanger.

        Returns
        -------
        dict
            The response with the survaval prediction and probability.
        """ 
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
        return {'Will the passanger survive?': self.survived[np.argmax(pred)],
                'Probability': str(int(round(max(pred[0]), 2)*100))+"%"}