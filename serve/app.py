import sys

from fastapi import FastAPI
from starlette.responses import JSONResponse

sys.path.append('../')
from titanic.titanic_classifier import TitanicClassifier
from titanic.utils.models import Passanger

app = FastAPI()
titanic_classifier = TitanicClassifier()

@app.get('/titanic/healthcheck', status_code=200)
async def healthcheck():
    return 'Titanic has set sail and is ready to sink!'

@app.on_event("startup")
async def startup():
    titanic_classifier.train()

@app.post('/titanic/sink_titanic')
def extract_name(passanger_features: Passanger):
    print(type(titanic_classifier))
    return JSONResponse(titanic_classifier.sink_titanic(passanger_features))