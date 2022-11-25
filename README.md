# titanic

This project creates  calssifier that predicts if a specific Titanic passanger will survive the sinking of it.
The classifier is served as an API in the local machine using FastApi Framework.

## How to use it?

### 1. Install requirements in a virtual environment

### 2. Open a terminal in the main folder and change to the folder titanic/serve

### 3. With the terminal, open the application using the following command:

<code> uvicorn app:app --reload </code> 

### 4. Open the local host on a browser with the documentation of fastapi

<code> http://127.0.0.1:8000/docs </cod>

### 5. Try out the POST/titanic/sink_titanic method and complete the dictionary with the passanger data:

<code> {
  "name": "Master Yahir",
  "sex": "male",
  "age": 26,
  "pclass": 1,
  "sibsp": 2,
  "parch": 2,
  "fare": 500,
  "cabin": "C202",
  "embarked": "S"
}
</code>

### 6. Click on the "EXECUTE" button and review the prediction results.
