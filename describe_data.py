import pathlib
import pandas as pd
import json
import mlflow
from toolz.itertoolz import pluck

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

# extract distribution
for name, dtype in data_df.dtypes.iteritems():
    if dtype==object:
        mlflow.set_experiment(experiment_name='EDA cat')
    else:
        mlflow.set_experiment(experiment_name='EDA num')

    with mlflow.start_run(run_name=name):
        stats = data_df[name].describe(include='all')
        stats['dtype'] = str(dtype)
        stats['desc'] = description_mapper.get(name)
        stats.index = stats.index.str.replace(pat='%', repl='pct')
        mlflow.log_params(stats.to_dict())

