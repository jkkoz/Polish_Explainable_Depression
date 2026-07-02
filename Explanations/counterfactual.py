# code adapted from https://github.com/interpretml/DiCE/tree/main/docs/source/notebooks

import dice_ml
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline  
from sklearn.metrics import accuracy_score

backend = 'sklearn'
RANDOM_STATE = 42

df = pd.read_csv('feature_vector.csv')

mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
df['sentiment'] = df['sentiment'].map(mapping)

features = ['n_1st_verb','n_1st_pronoun','n_sing','n_words', 'n_cond', 'n_past',
            'rate_cond','rate_verb','rate_pron','rate_sing', 'rate_past',
            'emoji','sentiment','absolutist']

df = df.dropna(subset=features)
df = df[['label'] + features]
target = df['label']

train_dataset, test_dataset, Y_train, Y_test = train_test_split(
    df, target, test_size=0.10, stratify=target, random_state=RANDOM_STATE
)

X_train = train_dataset.drop('label', axis=1)
X_test  = test_dataset.drop('label', axis=1)

d = dice_ml.Data(
    dataframe=train_dataset,
    continuous_features=['rate_verb','rate_pron','rate_sing','rate_cond', 'rate_past',
                         'n_words','n_1st_verb','n_1st_pronoun','n_sing', 'n_cond', 'n_past'
                         ],
    outcome_name='label'
)

numerical   = ['rate_verb','rate_pron','rate_sing', 'rate_cond', 'rate_past', 'n_past', 'n_cond','n_words','n_1st_verb','n_1st_pronoun','n_sing']
categorical = X_train.columns.difference(numerical)

transformations = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
    ('num', StandardScaler(), numerical)
])

logreg = LogisticRegression(
    solver="saga", 
    penalty="l2", 
    C=1.0,
    max_iter=2000, 
    n_jobs=-1, 
    random_state=RANDOM_STATE,
)

svm_linear = LinearSVC(C=1.0, random_state=RANDOM_STATE)
svm = CalibratedClassifierCV(svm_linear, method="sigmoid", cv=5)

clf = Pipeline(steps=[
    ('preprocessor', transformations),
    ('smote', SMOTE(sampling_strategy='minority', random_state=RANDOM_STATE)),
    ('classifier', logreg)
])

model = clf.fit(X_train, Y_train)

pred = model.predict(X_test)
acc = accuracy_score(Y_test, pred)
print("Accuracy:", acc)

m   = dice_ml.Model(model=model, backend=backend)
exp = dice_ml.Dice(d, m, method="kdtree")

e1 = exp.generate_counterfactuals(
    X_test[0:1],
    total_CFs=2,
    desired_class="opposite",
    permitted_range={
        'rate_verb': [0, 1],
        'rate_pron': [0, 1],
        'rate_sing': [0, 1],
        'rate_cond': [0, 1],
        'rate_past': [0, 1],
        'n_1st_verb' : [0,11],
        'n_1st_pronoun': [0,8],
        'n_sing': [0,11],
        'n_words': [4,323],
        'n_cond':[0,3],
        'n_past':[0,8]
    },
    features_to_vary={'n_1st_verb', 'rate_verb', 'n_words','sentiment', 'emoji', 'absolutist',
                      'n_sing', 'rate_sing',
                    #   'n_1st_pronoun','rate_pron',
                        'n_cond', 'n_past' , 'rate_cond', 'rate_past'
                    # 'n_sing', 'n_cond', 'n_past', 'rate_sing', 'rate_sing','rate_cond', 'rate_past',
                      }
)

print(e1.visualize_as_dataframe(show_only_changes=True))

query_instances = X_test
imp = exp.global_feature_importance(query_instances)
print(imp.summary_importance)
