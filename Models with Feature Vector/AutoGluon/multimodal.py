import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.multimodal import MultiModalPredictor
from sklearn.model_selection import train_test_split
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
from autogluon.tabular import FeatureMetadata
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_float32_matmul_precision('high')

df = TabularDataset('feature_vector.csv')

df['emoji'] = df['emoji'].astype(bool)
df['absolutist'] = df['absolutist'].astype(bool)


# change to check the performance with different subset of features
# features = ['n_1st_verb','n_sing','n_past','n_words', 'n_cond','n_1st_pronoun',
#             'rate_verb','rate_pron','rate_sing','rate_past','rate_cond', 
#             'emoji', 'sentiment', 'absolutist'
#             ] 
# df = df[features + ['label']]

df = df.dropna()

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

df_train.to_csv("feature_train_multi.csv", index=False)
df_val.to_csv("feature_val_multi.csv", index=False)
df_test.to_csv("feature_test_multi.csv", index=False)

train_data = TabularDataset(df_train)  
val_data = TabularDataset(df_val)

feature_metadata = FeatureMetadata.from_df(train_data)
feature_metadata = feature_metadata.add_special_types({
    'sentiment':  ['category'],   
    'emoji':      ['category'],  
    'absolutist': ['category'], 
    'text':       ['text'],      
})


hyperparameters = {
    'GBM': [
        {'device': 'cpu'},{'device': 'cpu', 'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}},
        {
            'device': 'cpu',
            "learning_rate": 0.03,
            "num_leaves": 128,
            "feature_fraction": 0.9,
            "min_data_in_leaf": 3,
            "ag_args": {"name_suffix": "Large", "priority": 0, "hyperparameter_tune_kwargs": None},
    }],
    'CAT': {'task_type': 'CPU'},
    'XGB': {'device': 'cpu'},  
    'AG_AUTOMM': [{                                                            # general light configuration because of machine constraints
        'model.hf_text.checkpoint_name': 'google/electra-small-discriminator', # for medium quality faster train; or distilroberta-base or allegro/herbert-base-cased
        'env.per_gpu_batch_size': 1,
        'optim.max_epochs': 5,
        'optim.patience': 4,
        'model.hf_text.gradient_checkpointing': True,
        'ag_args_fit': {'num_gpus': 1},
    }],
 }

predictor = TabularPredictor(label='label', eval_metric='f1',
                             path='AutogluonModels/multimodal_herbertf1', problem_type='binary').fit(
                                train_data=train_data, 
                                tuning_data=val_data,
                                feature_metadata=feature_metadata,
                                fit_strategy="sequential",
                                hyperparameters=hyperparameters,
                                presets='medium',
                                ag_args_fit={'num_gpus': 0}
                            )

test_data = TabularDataset(df_test)

y_pred = predictor.predict(test_data.drop(columns=['label'], axis=1))
print(y_pred.head())

print(predictor.evaluate(test_data))
print(predictor.leaderboard(test_data,extra_info=True, extra_metrics=['precision', 'recall' ,'f1','roc_auc'])[['model', 'score_test', 'score_val', 'eval_metric','num_features', 'precision', 'recall' ,'f1','roc_auc']])
print(predictor.feature_importance(data=test_data))
# print(feature_metadata)
# print(hyperparameters)
