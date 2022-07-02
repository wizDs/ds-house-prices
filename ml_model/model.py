from dataclasses import dataclass
from operator import attrgetter
import pathlib
from typing import Iterable
import numpy as np
import pandas as pd
import json
import mlflow
from toolz.itertoolz import pluck
from sklearn.compose import make_column_selector, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearndf.pipeline import PipelineDF
from sklearndf.transformation import StandardScalerDF, SimpleImputerDF
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from ml_model.eval_regression import ModelReport, ModelReportBuilder


num_ppl = PipelineDF(
    steps=[
        ('scaler', StandardScalerDF()),
        ('imputer', SimpleImputerDF(fill_value=-1)),
        # TODO: Outlier detection
    ]
)

model = Pipeline(
    steps=[
        ('preprocessor', num_ppl),
        ('model', LinearRegression(n_jobs=-1)
        
        
        )
    ]
)

tt = TransformedTargetRegressor(regressor=LinearRegression(n_jobs=-1),
                                 func=np.log, inverse_func=np.exp)

model_log = Pipeline(
    steps=[
        ('preprocessor', num_ppl),
        ('model', tt)
    ]
)


if __name__ == '__main__':

    curr_path = pathlib.Path('.')
    data_path = curr_path / 'data' / 'train.csv'

    # read data
    data_df = pd.read_csv(data_path)
    data_df.set_index('Id', inplace=True)

    # read feature descriptions
    with open(curr_path / 'data' / 'feature_description.txt', 'r') as f:
        feature_descriptions = json.loads(f.read())['features']

    # mapper from name to description
    description_mapper = dict(pluck(['name','desc'], feature_descriptions))
    
    num_cols = make_column_selector(dtype_include=np.number)
    bool_cols = make_column_selector(dtype_include=bool)

    X,y = data_df.drop(columns='SalePrice'), data_df['SalePrice']
    X[num_cols].pipe(num_ppl.fit_transform).describe().transpose().round(2)

    kfold = KFold(random_state=423, shuffle=True)
    train_index, test_index = next(kfold.split(X[num_cols]))
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model_log.fit(X_train[num_cols], y_train)
    y_pred = model_log.predict(X_test[num_cols])
    
    model_report = ModelReportBuilder(X[num_cols], y / 1_000_000, model, kfold)
    model_report.summary.dict()
    
    model_report = ModelReportBuilder(X[num_cols], y / 1_000_000, model_log, kfold)
    model_report.summary.dict()
    
    # test data
    test_path = curr_path / 'data' / 'test.csv'
    test_df = pd.read_csv(test_path)
    test_df.set_index('Id', inplace=True)

    model.predict(test_df[num_cols])