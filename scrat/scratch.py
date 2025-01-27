import pandas as pd
import pmlb

from howso.engine import (
    load_trainee,
    Trainee,
)
from howso.utilities import infer_feature_attributes


df = pmlb.fetch_data('iris')
features = infer_feature_attributes(df)

t = Trainee(name='Testing!!', features=features)

t.train(df)
t.analyze()
t.react_into_trainee(residuals=True)

stats = t.get_prediction_stats(robust=False)
print(stats)

stats = t.get_prediction_stats(stats=['confusion_matrix'], robust=False)
print(stats)

file_path = "./test_trainee.caml"

t.save(file_path=file_path)

t = load_trainee(file_path=file_path)

t.delete()
