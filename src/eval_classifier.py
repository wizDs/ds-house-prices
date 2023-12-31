from operator import attrgetter
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from typing import Callable, Iterable, List, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, validator
from sklearn.metrics import auc, roc_curve, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import KFold
from statistics import mean, stdev
from sklearn.base import clone, BaseEstimator

class ConfusionMatrix(BaseModel):
    tn: int
    fp: int
    fn: int
    tp: int

class ModelEvaluation(BaseModel):
    accuracy: float
    stu_precision: float
    stu_recall: float
    ltu_precision: float
    ltu_recall: float
    auc: Optional[float] = None
    confusion_matrix: Optional[ConfusionMatrix] = None


@dataclass
class ModelReport(object):
    model_reports: List[ModelEvaluation] = field(repr=False)
    avg_accuracy: float = field(init=False)
    std_accuracy: float = field(init=False)
    stu_avg_precision: float = field(init=False)
    stu_avg_recall: float = field(init=False)
    ltu_avg_precision: float = field(init=False, repr=False)
    ltu_avg_recall: float = field(init=False, repr=False)
    avg_auc: float = field(init=False)
    confusion_matrix: ConfusionMatrix = field(init=False)

    def __post_init__(self):
        best_model              = max(self.model_reports, key=attrgetter("auc"))
        self.avg_accuracy       = mean(r.accuracy for r in self.model_reports) 
        self.std_accuracy       = stdev(r.accuracy for r in self.model_reports) if len(self.model_reports) > 1 else None
        self.stu_avg_precision  = mean(r.stu_precision for r in self.model_reports)
        self.stu_avg_recall     = mean(r.stu_recall for r in self.model_reports)
        self.ltu_avg_precision  = mean(r.ltu_precision for r in self.model_reports)
        self.ltu_avg_recall     = mean(r.ltu_recall for r in self.model_reports)
        self.avg_auc            = mean(r.auc for r in self.model_reports) if sum(1 if r.auc else 0 for r in self.model_reports) else None
        self.confusion_matrix   = best_model.confusion_matrix

    def dict(self):
        return {k:v for k,v in self.__dict__.items() if (k not in ("model_reports", "confusion_matrix")) and ( v is not None)}

class ClassificationReportBuilder(ModelReport):

    def __init__(self, X: pd.DataFrame, y: pd.Series, model: Pipeline, kfold: KFold=KFold(n_splits=10, shuffle=True), transform_output: Optional[Callable]=None):
        reports = [self.eval_model(X, y, model, train_index, test_index, transform_output) for train_index, test_index in tqdm(kfold.split(X,y), total=kfold.n_splits)]
        super().__init__(model_reports=reports)
    
    @classmethod
    def eval_model(self, X: pd.DataFrame, y: pd.Series, model: Pipeline, train_index: Iterable[int], test_index: Iterable[int], transform_output: Optional[Callable]) -> ModelEvaluation:
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if transform_output:
            y_pred = transform_output(y_pred)
            y_test = transform_output(y_test)

        report = ModelEvaluation(
            accuracy        = accuracy_score(y_test, y_pred),
            stu_precision   = precision_score(y_test, y_pred, pos_label=1, zero_division=0),
            stu_recall      = recall_score(y_test, y_pred, pos_label=1),
            ltu_precision   = precision_score(y_test, y_pred, pos_label=0, zero_division=0),
            ltu_recall      = recall_score(y_test, y_pred, pos_label=0),
            auc             = self.get_auc(X_test, y_test, model), 
            confusion_matrix= self.get_confusion_matrix(y_test, y_pred),
        )
        return report
    
    @staticmethod
    def get_auc(X_test: pd.DataFrame, y_test: pd.Series, model: Pipeline) -> Optional[float]:
        """
        Wrapper for auc method from scikit learn 
        """
        # if model does not have predict_proba as method 
        if not (hasattr(model, "predict_proba") and callable(getattr(model, "predict_proba"))):
            return None
        
        y_proba = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        return auc(fpr, tpr)
    
    @staticmethod
    def get_confusion_matrix(y_test: pd.Series, y_pred: pd.Series) -> ConfusionMatrix:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        return ConfusionMatrix(tn=tn, fp=fp, fn=fn, tp=tp)
