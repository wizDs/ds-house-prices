from dataclasses import dataclass
import xgboost
import pathlib
import math
from typing import Callable, Iterable
import numpy as np
import pandas as pd
import json
import mlflow
from functools import partial
from toolz.itertoolz import pluck
from sklearn.compose import ColumnTransformer, make_column_selector, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from eval_regression import ModelReport, ModelReportBuilder

# a frunction that returns an esimator, that is a full ds-model
ModelConstructor = Callable[[None], BaseEstimator]
PreprocessorConstructor = Callable[[None], TransformerMixin]


def create_preprocessor_oe() -> ColumnTransformer:
    num_cols = make_column_selector(dtype_include=np.number)
    cat_cols = make_column_selector(dtype_include=object)
    num_ppl = Pipeline(steps=[('imputer', (SimpleImputer(fill_value=-1)
                                           .set_output(transform="pandas"))),
                              ('scaler', (StandardScaler()
                                          .set_output(transform="pandas"))),
                              ])
    # TODO: Outlier detection
    cat_ppl = Pipeline(steps=[('oe', (OrdinalEncoder(handle_unknown='use_encoded_value',
                                                     unknown_value=-1,
                                                     encoded_missing_value=-1)
                                      .set_output(transform='pandas'))),
                              # ('scaler', (StandardScaler()
                              #             .set_output(transform="pandas"))),
                              ])
    return (ColumnTransformer(transformers=[('num', num_ppl, num_cols),
                                            ('cat', cat_ppl, cat_cols),],
                              verbose_feature_names_out=False)
            .set_output(transform='pandas'))


def create_preprocessor_ohe() -> ColumnTransformer:
    num_cols = make_column_selector(dtype_include=np.number)
    cat_cols = make_column_selector(dtype_include=object)
    num_ppl = Pipeline(steps=[('imputer', (SimpleImputer(fill_value=-1)
                                           .set_output(transform="pandas"))),
                              ('scaler', (StandardScaler()
                                          .set_output(transform="pandas"))),
                              ])
    # TODO: Outlier detection
    cat_ppl = Pipeline(steps=[('ohe', (OneHotEncoder(handle_unknown='infrequent_if_exist',
                                                     drop='first',
                                                     sparse_output=False)
                                       .set_output(transform='pandas'))),
                              # ('scaler', (StandardScaler()
                              #             .set_output(transform="pandas"))),
                              ])
    return (ColumnTransformer(transformers=[('num', num_ppl, num_cols),
                                            ('cat', cat_ppl, cat_cols),],
                              verbose_feature_names_out=False)
            .set_output(transform='pandas'))


def create_xgboost(preprocessor: PreprocessorConstructor = None) -> Pipeline:
    '''gets a new instance of a xgboost model'''
    if not preprocessor:
        preprocessor = create_preprocessor_ohe
    return Pipeline(steps=[('preprocessor', preprocessor()),
                           ('model', xgboost.XGBRegressor())])


def create_log_linear(preprocessor: PreprocessorConstructor = None) -> Pipeline:
    @np.vectorize
    def truncated_exp(x: float) -> float:
        MIN_PRICE = 0
        MAX_PRICE = 600_000
        return max(min(math.exp(x), MAX_PRICE), MIN_PRICE)

    if not preprocessor:
        preprocessor = create_preprocessor_ohe

    tt_linear = TransformedTargetRegressor(regressor=LinearRegression(n_jobs=-1),
                                           func=np.log,
                                           inverse_func=truncated_exp,
                                           check_inverse=False)

    return Pipeline(steps=[('preprocessor', preprocessor()),
                           ('model', tt_linear)])


def create_knn(preprocessor: PreprocessorConstructor = None, **kwargs) -> Pipeline:
    if not preprocessor:
        preprocessor = create_preprocessor_ohe

    return Pipeline(steps=[('preprocessor', preprocessor()),
                           ('model', KNeighborsRegressor(**kwargs))])


def create_lasso(preprocessor: PreprocessorConstructor = None) -> Pipeline:
    if not preprocessor:
        preprocessor = create_preprocessor_ohe

    return Pipeline(steps=[('preprocessor', preprocessor()),
                           ('model', Lasso(tol=1e-3))])


def error_by_actual_price(X: pd.DataFrame,
                          y: pd.DataFrame,
                          model: BaseEstimator,
                          kfold: KFold,
                          error_measure: Callable = None) -> pd.DataFrame:
    '''get table with prediction error by actual price'''
    def mean_absolute_error(y_test, y_pred):
        return np.abs(y_test - y_pred)
    if not error_measure:
        error_measure = mean_absolute_error
    train_index, test_index = next(kfold.split(X))

    # evaluate model
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    eval_df = pd.DataFrame({'price': y_test,
                            'pred_price': y_pred.round(),
                            'error_pred': error_measure(y_test, y_pred).round()})

    return (eval_df
            .groupby(pd.cut(x=eval_df['price'] / 1_000,
                            bins=range(0, 650, 20)))
            .agg(error_pred=('error_pred', 'mean'),
                 std_pred=('error_pred', 'std'),
                 count=('error_pred', 'count'),)
            .round(0)
            .astype('Int64')
            .loc[lambda x: x['count'] > 0])


@dataclass
class EvaluationSet:
    features: pd.DataFrame
    labels: pd.Series
    model_constructor: ModelConstructor


def evaluate_multiple_models(k_fold: KFold, /, *args) -> Iterable[ModelReport]:
    for eval_set in args:
        report = ModelReportBuilder(eval_set.features,
                                    eval_set.labels,
                                    eval_set.model_constructor(),
                                    k_fold)
        yield report


if __name__ == '__main__':

    num_cols = make_column_selector(dtype_include=np.number)
    cat_cols = make_column_selector(dtype_include=object)

    curr_path = pathlib.Path('.')
    data_path = curr_path / 'data' / 'train.csv'

    # read data
    data_df = pd.read_csv(data_path)
    data_df.set_index('Id', inplace=True)

    # read feature descriptions
    with open(curr_path / 'data' / 'feature_description.txt', 'r', encoding='utf-8') as f:
        feature_descriptions = json.loads(f.read())['features']

    # mapper from name to description
    description_mapper = dict(pluck(['name', 'desc'], feature_descriptions))

    X, y = data_df.drop(columns='SalePrice'), data_df['SalePrice']
    X = X.rename(columns=lambda s: s.lower().replace(
        '.', '_').replace('(', '').replace(')', ''))
    for col in X[cat_cols].columns:
        X[col] = X[col].fillna('')
    # X[num_cols].pipe(num_ppl.fit_transform).describe().transpose().round(2)

    kfold = KFold(random_state=423, shuffle=True)
    ksplit = kfold.split(X)

    distr_df = error_by_actual_price(X, y, create_xgboost(), kfold)
    print(distr_df)

    evaluation_set = [
        EvaluationSet(features=X[num_cols],
                      labels=y,
                      model_constructor=create_xgboost),
        EvaluationSet(features=X,
                      labels=y,
                      model_constructor=create_xgboost),
        EvaluationSet(features=X,
                      labels=y,
                      model_constructor=partial(create_xgboost,
                                                preprocessor=create_preprocessor_oe)),
        EvaluationSet(features=X[num_cols],
                      labels=y,
                      model_constructor=partial(create_log_linear,
                                                preprocessor=create_preprocessor_oe)),
        EvaluationSet(features=X[X[num_cols].columns.tolist() +
                                 ['poolqc', 'fence', 'paveddrive']],
                      labels=y,
                      model_constructor=partial(create_log_linear,
                                                preprocessor=create_preprocessor_oe)),
        EvaluationSet(features=X[X[num_cols].columns.tolist() +
                                 ['poolqc', 'fence', 'paveddrive']],
                      labels=y,
                      model_constructor=create_log_linear),
        EvaluationSet(features=X[num_cols],
                      labels=y,
                      model_constructor=create_lasso),
        EvaluationSet(features=X,
                      labels=y,
                      model_constructor=create_knn),
        EvaluationSet(features=X,
                      labels=y,
                      model_constructor=partial(create_knn,
                                                preprocessor=create_preprocessor_oe)),
    ]

    reports = evaluate_multiple_models(kfold, *evaluation_set)
    for eval_set, r in zip(evaluation_set, reports):
        print('\n\n')
        print(eval_set.model_constructor
              if isinstance(eval_set.model_constructor, partial)
              else eval_set.model_constructor.__name__)
        print('column counts: ',
              len(eval_set.features.columns.tolist()))
        print(json.dumps(r.summary.dict(), indent=4))

    # test data
    test_path = curr_path / 'data' / 'test.csv'
    test_df = pd.read_csv(test_path)
    test_df.set_index('Id', inplace=True)
    X_validation = (test_df
                    .rename(columns=lambda s: (s
                                               .lower()
                                               .replace('.', '_')
                                               .replace('(', '')
                                               .replace(')', ''))))

    model = create_xgboost()
    model.fit(X, y)
    y_validation_pred = model.predict(X_validation).round().astype(int)
    y_validation_pred
