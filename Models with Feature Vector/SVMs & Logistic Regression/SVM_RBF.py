import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('feature_vector.csv')

# map sentiment to integers for standarization purposes
senti = ('positive', 'neutral', 'negative')
senti_map = (1, 0, -1)

mapping = dict(zip(senti, senti_map))
df['sentiment'] = df['sentiment'].map(mapping)

# specify the relevant features
features = ['n_1st_verb','n_1st_pronoun','n_sing','n_past','n_words',
            'rate_verb','rate_pron','rate_sing','rate_past','emoji', 'sentiment',
            #'rate_cond', 'n_cond'
            ]

df = df.dropna(subset=features)

X = df[features].values
Y = df['label'].values

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.10, stratify=Y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test) 

# oversample the minority class (depression tweets)
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, Y_train)

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
              'gamma':[0.0001, 0.001, 0.01]}

# perform grid search
svm = SVC(kernel='rbf')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(svm, 
                           param_grid, 
                           cv=cv, 
                           n_jobs=-1,
                           scoring='f1')
grid_search.fit(X_train_sm, y_train_sm)

print(
    "Best parameters are {} \nScore : {}%".format(
        grid_search.best_params_, grid_search.best_score_*100)
)

# test on the the test set
Y_pred = grid_search.predict(X_test)
print("\nTest set report:")
print(classification_report(Y_test, Y_pred))
