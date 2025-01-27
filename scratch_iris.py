import numpy as np
import pandas as pd
import pmlb

from howso.engine import Trainee
from howso.utilities import infer_feature_attributes

df = pmlb.fetch_data("iris")

features = infer_feature_attributes(df)

t = Trainee(features=features)


t.train(df)
t.analyze()

mat = t.get_contribution_matrix(targeted=False, robust=False)
print(mat)

t.save()