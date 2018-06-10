import requests
import pandas as pd
import json

api_base_url = 'http://localhost:5000/predictions'

head = {'Content-Type': 'application/json'}

filename = '/Users/jacobhumber/python_projects/simple_model_api/ml_model/test_head.csv'

d = pd.read_csv(filename)

d_json = json.dumps(d.to_json(orient = 'columns'))

r = requests.post(api_base_url, headers = head, json = d_json)

print(r.json())





'''
r_get = requests.get(api_base_url, headers = head)

print(r_get.content.decode('utf-8'))

r = requests.post(api_base_url, headers = head, json = {'fudge':'fuck'})

print(r.content.decode('utf-8'))

r_get = requests.get(api_base_url, headers = head)

print(r_get.content.decode('utf-8'))
'''