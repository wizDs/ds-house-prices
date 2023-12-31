from operator import attrgetter
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from typing import Callable, Iterable, List, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel
from sklearn.metrics import auc, roc_curve, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import KFold
from statistics import mean, stdev
from sklearn.metrics import (
    explained_variance_score, 
    max_error, 
    mean_absolute_error, 
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

class ModelEvaluation(BaseModel):
    explained_variance_score: float
    max_error: float
    mean_absolute_error: float
    mean_squared_error: float
    mean_absolute_percentage_error: float
    r2_score: float

class ModelReport(BaseModel):
    evaluations: List[ModelEvaluation]

    @property
    def summary(self) -> ModelEvaluation:
        return ModelEvaluation(
            explained_variance_score =      mean(self.get_attr('explained_variance_score')),
            max_error =                     mean(self.get_attr('max_error')),
            mean_absolute_error =           mean(self.get_attr('mean_absolute_error')),
            mean_squared_error =            mean(self.get_attr('mean_squared_error')),
            mean_absolute_percentage_error= mean(self.get_attr('mean_absolute_percentage_error')),
            r2_score =                      mean(self.get_attr('r2_score')),
        )

    def get_attr(self, attr: str) -> Iterable[float]:
        return map(attrgetter(attr), self.evaluations)
    
class ModelReportBuilder(ModelReport):

    def __init__(self, X: pd.DataFrame, y: pd.Series, model: Pipeline, kfold: KFold=KFold(n_splits=10, shuffle=True)):
        evaluations = [self.eval_model(X, y, model, train_index, test_index) for train_index, test_index in tqdm(kfold.split(X,y), total=kfold.n_splits)]
        super().__init__(evaluations=evaluations)
    
    @classmethod
    def eval_model(self, X: pd.DataFrame, y: pd.Series, model: Pipeline, train_index: Iterable[int], test_index: Iterable[int]) -> ModelEvaluation:
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return ModelEvaluation(
            explained_variance_score = explained_variance_score(y_test, y_pred),
            r2_score = r2_score(y_test, y_pred),
            max_error = max_error(y_test, y_pred),
            mean_absolute_error = mean_absolute_error(y_test, y_pred),
            mean_squared_error = mean_squared_error(y_test, y_pred),
            mean_absolute_percentage_error = mean_absolute_percentage_error(y_test, y_pred),
        )
        