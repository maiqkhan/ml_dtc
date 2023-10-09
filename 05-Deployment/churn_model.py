# %% [markdown]
# ## 4. Evaluation Metrics for Classification

import pickle
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

C = 1.0
output_file = f'model_C={C}.bin'
n_splits = 5

# %%
df = pd.read_csv('https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# %%
df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)

# %%
df_full_train, df_test =  train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.churn.values
y_val = df_val.churn.values 
y_test = df_test.churn.values 

del df_train['churn']
del df_val['churn']
del df_test['churn']


# %%
numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
        'phoneservice', 'multiplelines', 'internetservice',
       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
       'paymentmethod']

# %%
dv = DictVectorizer(sparse=False)

train_dicts = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

model = LogisticRegression()

model.fit(X_train, y_train)


# %%
val_dicts = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dicts)

y_pred = model.predict_proba(X_val)[:, 1]
churn_decision = (y_pred >= 0.5)
(y_val == churn_decision).mean()

# %% [markdown]
# ### 4.2 Accuracy and Dummy Model

# %%
len(y_val)

# %%
(y_val == churn_decision).sum()

# %%
# change threshold

thresholds = np.linspace(0, 1, 21)
scores = []

for t in thresholds:
    churn_decision = (y_pred >= t)
    score = (y_val == churn_decision).mean()
    print(f"threshold {t} : score {score}")
    scores.append(score)



# %%
plt.plot(thresholds,scores)

# %%
from sklearn.metrics import accuracy_score

accuracy_score(y_val, y_pred >= 0.5)

# %%
from collections import Counter

Counter(y_pred >= 1.0)

# %%
Counter(y_val)   ##Class imbalance

# %% [markdown]
# ### 4.3 Confusion Table

# %%
actual_positive = (y_val == 1)
actual_negative = (y_val == 0)

# %%
t = 0.5
predict_positive = (y_pred >= t)
predict_negative = (y_pred < t)

# %%
tp = (predict_positive & actual_positive).sum()
tn = (predict_negative & actual_negative).sum()

# %%
fp = (predict_positive & actual_negative).sum()
fn = (predict_negative & actual_positive).sum()

# %%
confusion_matrix = np.array([
    [tn, fp],
    [fn, tp]
])
confusion_matrix

# %%
confusion_matrix / confusion_matrix.sum()


# %% [markdown]
# ### 4.4 Precision and Recall

# %%
(tp + tn) / (tp + tn + fp + fn)

# %%
#precision - tells how many positive predictiosn end up as correct
p = tp / (tp + fp)
p

# %%
#recall - tells how many positive ovservations end up being predicted as correct
r = tp / (tp + fn)
r

# %% [markdown]
# ### 4.5 ROC Curves

# %% [markdown]
# #### TPR and FRP

# %%
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)
tpr, fpr

# %%
thresholds = np.linspace(0, 1, 101)
scores = []

for t in thresholds:
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)

    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)

    tp = (predict_positive & actual_positive).sum()
    tn = (predict_negative & actual_negative).sum()

    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()

    scores.append((t, tp, fp, fn, tn))


# %%
columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
df_scores = pd.DataFrame(scores, columns=columns)

# %%
df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
df_scores['fpr']  = df_scores.fp / (df_scores.fp + df_scores.tn)

# %%
plt.plot(df_scores.threshold, df_scores['tpr'], label='TPR')
plt.plot(df_scores.threshold, df_scores['fpr'], label='FPR')
plt.legend()

# %% [markdown]
# #### Random Model

# %%
np.random.seed(1)
y_rand = np.random.uniform(0,1, size=len(y_val))

# %%
((y_rand >= 0.5) == y_val).mean()

# %%
def tpr_fpr_df(y_val, y_pred):
    thresholds = np.linspace(0, 1, 101)
    scores = []

    for t in thresholds:
        actual_positive = (y_val == 1)
        actual_negative = (y_val == 0)

        predict_positive = (y_pred >= t)
        predict_negative = (y_pred < t)

        tp = (predict_positive & actual_positive).sum()
        tn = (predict_negative & actual_negative).sum()

        fp = (predict_positive & actual_negative).sum()
        fn = (predict_negative & actual_positive).sum()

        scores.append((t, tp, fp, fn, tn))

    columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
    df_scores = pd.DataFrame(scores, columns=columns)

    df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
    df_scores['fpr']  = df_scores.fp / (df_scores.fp + df_scores.tn)

    return df_scores
        


# %%
df_rand = tpr_fpr_df(y_val, y_rand)

# %%
plt.plot(df_rand.threshold, df_rand['tpr'], label='TPR')
plt.plot(df_rand.threshold, df_rand['fpr'], label='FPR')
plt.legend()

# %% [markdown]
# Ideal model

# %%
num_neg = (y_val == 0).sum()
num_pos = (y_val == 1).sum()
num_neg, num_pos

# %%
y_ideal = np.repeat([0,1], [num_neg, num_pos])
y_ideal

# %%
y_ideal_pred = np.linspace(0,1, len(y_val))

# %%
((y_ideal_pred >= 0.726) == y_ideal).mean()

# %%
df_ideal = tpr_fpr_df(y_ideal, y_ideal_pred)

# %%
plt.plot(df_ideal.threshold, df_ideal['tpr'], label='TPR', color='black')
plt.plot(df_ideal.threshold, df_ideal['fpr'], label='FPR', color='black')
plt.legend()



# %% [markdown]
# Putting Everything Together

# %%
plt.plot(df_ideal.threshold, df_ideal['tpr'], label='TPR', color='black')
plt.plot(df_ideal.threshold, df_ideal['fpr'], label='FPR', color='black')
plt.legend()
# plt.plot(df_rand.threshold, df_rand['tpr'], label='TPR')
# plt.plot(df_rand.threshold, df_rand['fpr'], label='FPR')
# plt.legend()

plt.plot(df_scores.threshold, df_scores['tpr'], label='TPR')
plt.plot(df_scores.threshold, df_scores['fpr'], label='FPR')
plt.legend()

# %%
plt.figure(figsize=(6,6))

plt.plot(df_scores.fpr, df_scores.tpr, label='model')
plt.plot([0,1])
#plt.plot(df_rand.fpr, df_rand.tpr, label='random')
#plt.plot(df_ideal.fpr, df_ideal.tpr, label='ideal')

plt.xlabel('FPR')
plt.ylabel('TPR')

plt.legend()

# %%
#plot auc curves with sklearn

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_val, y_pred)

# %%
plt.figure(figsize=(6,6))

plt.plot(fpr, tpr, label='model')
plt.plot([0,1])
#plt.plot(df_rand.fpr, df_rand.tpr, label='random')
#plt.plot(df_ideal.fpr, df_ideal.tpr, label='ideal')

plt.xlabel('FPR')
plt.ylabel('TPR')

plt.legend()

# %% [markdown]
# ### 4.6 ROC AUC

# %%
from sklearn.metrics import auc

# %%
auc(fpr, tpr)

# %%
auc(df_scores.fpr, df_scores.tpr)

# %%
auc(df_ideal.fpr, df_ideal.tpr)

# %%
from sklearn.metrics import roc_auc_score

roc_auc_score(y_val, y_pred)

# %%
neg = y_pred[y_val == 0]
pos = y_pred[y_val == 1]

# %%
import random

n = 100_000
success = 0

for i in range(n):
    pos_ind = random.randint(0, len(pos) -1)
    neg_ind = random.randint(0, len(neg) -1)

    if pos[pos_ind] > neg[neg_ind]:
        success = success + 1

success / n

# %%
n = 10_000
pos_ind = np.random.randint(0, len(pos), size = n)
neg_ind = np.random.randint(0, len(neg), size = n)

# %%
(pos[pos_ind] > neg[neg_ind]).mean()

# %% [markdown]
# ### 4.7 Cross-Validation

# %%
def train(df, y, C=1.0):
    dicts = df[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y)


    return dv, model

# %%
dv, model = train(df_train, y_train)

# %%
def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)

    y_pred = model.predict_proba(X)[:, 1]
    return y_pred

# %%




print('doing validation with C = {C}')
for C in tqdm([0.001, 0.01, 0.1, 0.5, 1, 5, 10]):

    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    scores = []

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train.churn.values
        y_val = df_val.churn.values
        
        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print(f'C={C} {np.mean(scores)} +- {np.std(scores)}')


# %%
print('training the final model')
dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)
auc = roc_auc_score(y_test, y_pred)
auc

# %% [markdown]
# Save the model

# %%



output_file

# %%
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

# %% [markdown]
# Load the model

# %%

# %%
with open(output_file, 'rb') as f_in:
    (dv,model) = pickle.load(f_in)

# %%
dv, model

# %%
customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}

# %%
X = dv.transform([customer])

# %%
model.predict_proba(X)[0, 1]

# %%


# %% [markdown]
# 


