import pickle
import sklearn


model_file = r'model1.bin'
dv_file = r'dv.bin'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

with open(dv_file, 'rb') as f_in:
    dv = pickle.load(f_in)


X = dv.transform([{"job": "retired", "duration": 445, "poutcome": "success"}])

y_pred = model.predict_proba(X)[0,1]
print(y_pred)