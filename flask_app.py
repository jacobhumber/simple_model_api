from flask import Flask, request, jsonify 
import pandas as pd
from sklearn.externals import joblib
#from flask_debugtoolbar import DebugToolbarExtension
import json


app = Flask(__name__)

@app.route('/')
def index():
	return jsonify('hi1')

@app.route('/predictions')
def add_to_fake_stuff():
	return ('</body>hi1</body>')


@app.route('/predictions', methods=['POST'])
def gen_predict():
	#import pdb; pdb.set_trace()
	#handle data
	dtest_json = request.get_json()
	dtest = pd.read_json(json.loads(dtest_json), orient='records')

	#predict model
	filename = '/Users/jacobhumber/python_projects/simple_model_api/ml_model/simple_logit.pkl'
	model = joblib.load(filename)

	prediction = pd.DataFrame(model.predict(dtest))

	response = jsonify(prediction.to_json(orient="records"))

	return(response)


if __name__ == '__main__':
    app.run()	






