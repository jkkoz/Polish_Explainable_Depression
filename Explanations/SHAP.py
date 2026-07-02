# adapted from https://github.com/autogluon/autogluon/blob/master/examples/tabular/interpret/SHAP%20with%20AutoGluon-Tabular.ipynb

import matplotlib
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor
import pandas as pd
import numpy as np
np.int = int
np.float = float

import shap

model_path = ("/home/jkoz/bproject/AutogluonModels/inMethodsf1_no_text") # change to your path

test_data = pd.read_csv("feature_test.csv")
df_train = pd.read_csv('feature_train.csv')
df_val = pd.read_csv('feature_val.csv')

train_data = pd.concat([df_train, df_val], ignore_index=True)

features = ['n_1st_verb','n_sing','n_past','n_words', 'n_cond','n_1st_pronoun',
            'rate_verb','rate_pron','rate_sing','rate_past','rate_cond', 
            'emoji', 'sentiment', 'absolutist'
            ]

X_test = test_data[features]
y_test = test_data["label"]
X_train = train_data[features]
y_train = train_data["label"]


predictor = TabularPredictor.load(model_path)
print(predictor.leaderboard())

target_class = 1 # explain predictions of depressed

class AutogluonWrapper:
    def __init__(self, predictor, feature_names, target_class=None, model=None):
        self.ag_model = predictor
        self.feature_names = feature_names
        self.target_class = target_class
        self.model = model
        if target_class is None and predictor.problem_type != 'regression':
            print("Since target_class not specified, SHAP will explain predictions for each class")
    
    def predict_proba(self, X):
        if isinstance(X, pd.Series):
            X = X.values.reshape(1,-1)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        preds = self.ag_model.predict_proba(X, model=self.model)
        if self.ag_model.problem_type == "regression" or self.target_class is None:
            return preds
        else:
            return preds[self.target_class]    
        
negative_class = 0
NSHAP_SAMPLES = 200

baseline = X_train[y_train==negative_class].sample(NSHAP_SAMPLES, random_state=42)

ag_wrapper = AutogluonWrapper(predictor, X_train.columns, target_class, model='RandomForestEntr_BAG_L1')
explainer = shap.KernelExplainer(ag_wrapper.predict_proba, baseline)
print("Baseline prediction: ", np.mean(ag_wrapper.predict_proba(baseline)))  # this is the same as explainer.expected_value

shap.initjs()

# helper for user study construction:
# depressed: 296, 76, 94, 107, 406, 442, 328
# non depressed:  82, 129,298, 406
# new 388

ROW_INDEX = 74
single_datapoint = X_test.iloc[[ROW_INDEX]]

single_prediction = ag_wrapper.predict_proba(single_datapoint)

print(single_prediction)
print(predictor.predict(single_datapoint))
print(single_datapoint)

shap_values_single = explainer.shap_values(single_datapoint, nsamples=NSHAP_SAMPLES)

shap.force_plot(explainer.expected_value, shap_values_single,  X_test.iloc[ROW_INDEX, :], matplotlib=True, show=False)

plt.tight_layout()
plt.savefig("SHAP/inMethods.png", dpi=300, bbox_inches="tight")
plt.close()

shap_values = explainer.shap_values(X_test, nsamples=NSHAP_SAMPLES)
shap.summary_plot(shap_values, X_test)
plt.tight_layout()
plt.savefig("SHAP/summaryMethods_for_explpng", dpi=300, bbox_inches="tight")
plt.close()
