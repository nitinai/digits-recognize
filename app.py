__author__ = "Nitin Patil"

import os
import json
import base64
import random
from flask import Flask, render_template, request
from flask_cors import CORS
from models.cnn import *

app = Flask(__name__)
CORS(app, headers=['Content-Type'])

model = CNN()
model.load_state_dict(torch.load("models/model.pkl", map_location='cpu'))
model.eval()

@app.route("/", methods=["GET"])
def index_page():
	return render_template('index.html')
	

@app.route('/classify', methods = ["POST"])
def predict():
	"Decodes image and uses it to make prediction"
	if request.method == 'POST':
		image_b64 = request.values['imageBase64']
		image_encoded = image_b64.split(',')[1]
		image = base64.decodebytes(image_encoded.encode('utf-8'))
		prediction = single_predict(model,image)

	return json.dumps(prediction)


@app.route('/randomNum', methods = ["GET"])
def generateRandomNumber():
	prediction = random.randint(0, 100)
	return json.dumps(prediction)
		

if __name__ == '__main__':
	port = int(os.environ.get("PORT", 5000))
	app.run(host='0.0.0.0', port=port, debug=False)
