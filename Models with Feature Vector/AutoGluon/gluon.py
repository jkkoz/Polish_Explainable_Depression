import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor, FeatureMetadata

RANDOM_STATE = 42
TEST_SIZE = 0.10
VAL_SIZE = 0.10 

df_train = pd.read_csv("feature_train.csv")
df_val = pd.read_csv("feature_val.csv")
df_test = pd.read_csv("feature_test.csv")

features = ['n_1st_verb','n_sing','n_past','n_words', 'n_cond','n_1st_pronoun',
            'rate_verb','rate_pron','rate_sing','rate_past','rate_cond', 
            'emoji', 'sentiment', 'absolutist',
            ]

df_train = df_train[features + ['label']]
df_train = df_train.dropna()
df_val = df_val[features + ['label']]
df_val = df_val.dropna()
df_test = df_test[features + ['label']]
df_test = df_test.dropna()

train_data = TabularDataset(df_train)
val_data = TabularDataset(df_val)
test_data = TabularDataset(df_test)

feature_metadata = FeatureMetadata.from_df(train_data)
feature_metadata = feature_metadata.add_special_types({
    'sentiment':  ['category'],   
    'emoji':      ['category'],  
    'absolutist': ['category'], 
})

predictor = TabularPredictor(label='label', eval_metric='f1').fit(train_data=train_data, tuning_data=val_data, 
                                                                  use_bag_holdout=True, presets='good')
# alternatively change eval_metric='accuracy' and presets='medium'

y_pred = predictor.predict(test_data.drop(columns=['label'], axis=1))
print(y_pred.head())
print(predictor.evaluate(test_data))
print(predictor.leaderboard(test_data)) # get leaderboard
print(predictor.feature_importance(data=test_data)) # get feature importance

