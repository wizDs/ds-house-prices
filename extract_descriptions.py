import re
import pathlib
from typing import Optional
import pandas as pd
import pandas_profiling
import streamlit as st
from functools import partial
from pydantic import BaseModel
from streamlit_pandas_profiling import st_profile_report

class FeatureDescription(BaseModel):
    name: str
    desc: str

def read_feature_descriptions(path: pathlib.Path) -> pd.DataFrame:
    # read feature names from description file
    with open(path) as f:
        features = f.readlines()
        features = filter(lambda s: ":" in s, features)
        features = filter(partial(re.match, "\\w"), features)
        features = map(partial(re.sub, '\\n', ''), features)
        features = map(partial(re.sub, '\\t', ''), features)
        features = map(partial(re.split, ':\\s+?'), features)
        features = map(lambda x: FeatureDescription(name=x[0], desc=x[1]), features)
        features = pd.DataFrame(x.dict() for x in features)
        
    return features

def generate_streamlit_app(dataframes: list[pd.DataFrame], dataraport: Optional[pd.DataFrame] = None) -> None:
    for df in dataframes:
        st.table(df)

    if dataraport is not None:
        st_profile_report(dataraport.profile_report())

if __name__ == '__main__':
    
    curr_path = pathlib.Path('.')   
    feature_description = read_feature_descriptions(curr_path / 'data' / 'data_description.txt')
    train = pd.read_csv(curr_path / "data" / "train.csv")
    datatypes = train.dtypes.rename("datatype").rename_axis("name")
    data_summarization = train.describe(include='all').transpose()[['mean', 'std', 'min', 'max', 'unique', 'top', 'freq']]
    feature_description = feature_description \
        .join(
            datatypes, 
            on='name'
        ) \
        .join(
            data_summarization, 
            on='name'
        ) \
        .round(decimals=2)
    generate_streamlit_app(
        dataframes=[feature_description, train.sample(5)],
        dataraport=train
    )