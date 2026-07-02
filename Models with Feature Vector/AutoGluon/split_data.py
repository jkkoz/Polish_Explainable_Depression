import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split

df = pd.read_csv('feature_vector.csv')

RANDOM_STATE = 42
TEST_SIZE = 0.10
VAL_SIZE = 0.10 

df_trainval, df_test = train_test_split(
    df,
    test_size=TEST_SIZE,
    stratify=df["label"],
    random_state=RANDOM_STATE,
)

val_size_relative = VAL_SIZE / (1.0 - TEST_SIZE)  # 0.10 / 0.90 = 0.111...
df_train, df_val = train_test_split(
    df_trainval,
    test_size=val_size_relative,
    stratify=df_trainval["label"],
    random_state=RANDOM_STATE,
)


df_train.to_csv("feature_train.csv", index=False)
df_val.to_csv("feature_val.csv", index=False)
df_test.to_csv("feature_test.csv", index=False)
