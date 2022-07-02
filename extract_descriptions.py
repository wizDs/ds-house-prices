import re
import pathlib
from functools import partial
from pydantic import BaseModel
from typing import List

curr_path = pathlib.Path('.')

class FeatureDescription(BaseModel):
    name: str
    desc: str

class FeatureDescriptions(BaseModel):
    features: List[FeatureDescription] 



# read feature names from description file
with open(curr_path / 'data' / 'data_description.txt') as f:
    features = f.readlines()
    features = filter(lambda s: ":" in s, features)
    features = filter(partial(re.match, "\\w"), features)
    features = map(partial(re.sub, '\\n', ''), features)
    features = map(partial(re.sub, '\\t', ''), features)
    features = map(partial(re.split, ':\\s+?'), features)
    features = map(lambda x: FeatureDescription(name=x[0], desc=x[1]), features)
    features = FeatureDescriptions(features=list(features))


with open(curr_path / 'data' / 'feature_description.txt', 'w') as f:
    f.write(features.json())
