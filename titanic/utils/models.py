#Python
from typing import Optional
from enum import Enum

#Pydantic
from pydantic import BaseModel
from pydantic import Field

class Sex(Enum):
    female= "female"
    male= "male"

    def __str__(self)->str:
        return str(self.value)

class Embarked(Enum):
    S ='S' 
    C = 'C'
    Q = 'Q'

    def __str__(self)->str:
        return str(self.value)

class Passanger(BaseModel):
    name: str = Field(
        ...,
        min_length = 1,
        max_length=50,
        example="Master Yahir"
        )
    sex: Sex = Field(
        ...,
        example = 'male'
        )
    age: int = Field(
        ...,
        ge=0,
        le=100,
        example = 26
        )
    pclass: int = Field(
        ...,
        ge = 1,
        le = 3,
        example = 1
        )
    sibsp: int = Field(
        ...,
        ge = 0,
        le = 8,
        example = 2
        )
    parch: int = Field(
        ...,
        ge = 0,
        le = 9,
        example = 2
        )
    fare: float = Field(
        ...,
        ge = 0,
        le = 600,
        example = 500
        )
    cabin: Optional[str] = Field(
        example = 'C202'
        )
    embarked: Optional[Embarked] = Field(
        example = 'S'
        )
    