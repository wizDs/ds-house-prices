import xgboost
import pathlib
import math
from typing import Iterable
import numpy as np
import pandas as pd
import json
import mlflow
from toolz.itertoolz import pluck
from sklearn.compose import ColumnTransformer, make_column_selector, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso

from sklearn.model_selection import KFold
from ml_model.eval_regression import ModelReportBuilder


num_cols = make_column_selector(dtype_include=np.number)
cat_cols = make_column_selector(dtype_include=object)
num_ppl = Pipeline(steps=[('scaler', (StandardScaler()
                                      .set_output(transform="pandas"))),
                          ('imputer', (SimpleImputer(fill_value=-1)
                                       .set_output(transform="pandas"))),])
                          # TODO: Outlier detection
cat_ppl = Pipeline(steps=[('ohe', (OneHotEncoder(handle_unknown='infrequent_if_exist',
                                                  sparse_output=False)
                                   .set_output(transform='pandas'))),
                          ('scaler', (StandardScaler()
                                      .set_output(transform="pandas"))),
                          ])
preprocessor = (ColumnTransformer(transformers=[('num', num_ppl, num_cols),
                                                ('cat', cat_ppl, cat_cols),])
                .set_output(transform='pandas'))
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('model', xgboost.XGBRegressor())])

@np.vectorize
def truncated_exp(x: float) -> float:
    MIN_PRICE = 0
    MAX_PRICE = 600_000
    return float(max(min(math.exp(x), MAX_PRICE), MIN_PRICE))

tt_linear = TransformedTargetRegressor(regressor=LinearRegression(n_jobs=-1),
                                        func=np.log,
                                        inverse_func=truncated_exp,
                                        check_inverse=False)

model_linear_log = Pipeline(steps=[('preprocessor', preprocessor),
                            ('model', tt_linear)])

if __name__ == '__main__':

    curr_path = pathlib.Path('.')
    data_path = curr_path / 'data' / 'train.csv'

    # read data
    data_df = pd.read_csv(data_path)
    data_df.set_index('Id', inplace=True)

    # read feature descriptions
    with open(curr_path / 'data' / 'feature_description.txt', 'r', encoding='utf-8') as f:
        feature_descriptions = json.loads(f.read())['features']

    # mapper from name to description
    description_mapper = dict(pluck(['name','desc'], feature_descriptions))

    num_cols = make_column_selector(dtype_include=np.number)
    bool_cols = make_column_selector(dtype_include=bool)

    X,y = data_df.drop(columns='SalePrice'), data_df['SalePrice']
    X[num_cols].pipe(num_ppl.fit_transform).describe().transpose().round(2)

    kfold = KFold(random_state=423, shuffle=True)
    ksplit = kfold.split(X)
    next(ksplit)
    train_index, test_index = next(ksplit)

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    model_linear_log.fit(X_train[num_cols], y_train)
    y_pred_log = model_linear_log.predict(X_test[num_cols])

    # model_lasso.fit(X_train[num_cols], y_train)
    # y_pred_lasso = model_lasso.predict(X_test[num_cols])

    eval_df = pd.DataFrame({'price':y_test,
                            'pred_price':y_pred.round(),
                            'pred_price_log':y_pred_log.round()})
    eval_df['error_pred'] = np.abs(eval_df['pred_price'] - eval_df['price'])
    eval_df['error_pred_log'] = np.abs(eval_df['pred_price_log'] - eval_df['price'])

    distr_df = (eval_df
     .groupby(pd.cut(x=eval_df['price'] / 1_000, bins=range(0, 650, 20)))
     .agg(error_pred=('error_pred', 'mean'),
          error_pred_log=('error_pred_log', 'mean'),
          std_pred=('error_pred', 'std'),
          std_pred_log=('error_pred_log', 'std'),
          count=('error_pred_log', 'count'),)
     .round(0)
     .astype('Int64')
     .loc[lambda x: x['count'] > 0])

    model_report = ModelReportBuilder(X, y, model, kfold)
    print(json.dumps(model_report.summary.dict(), indent=4))

    model_report_log = ModelReportBuilder(X[num_cols], y, model_linear_log, kfold)
    print(json.dumps(model_report_log.summary.dict(), indent=4))

    # test data
    test_path = curr_path / 'data' / 'test.csv'
    test_df = pd.read_csv(test_path)
    test_df.set_index('Id', inplace=True)

    model.predict(test_df[num_cols])
