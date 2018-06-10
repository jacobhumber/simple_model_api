import pandas as pd
import json

filename = '/Users/jacobhumber/python_projects/simple_model_api/ml_model/test_head.csv'

d = pd.read_csv(filename)

d_json = json.dumps(d.to_json(orient = 'columns'))

#print()json.loads(d_json)

#dtest_json = request.get_json()
dtest = pd.read_json(json.loads(d_json), orient='records')
'''
#predict model
filename = '../ml_model/simple_logit.pkl'
model = joblib.load(filename)

prediction = model.predict(dtest)

response = jsonify(prediction.to_json(orient="records"))
'''