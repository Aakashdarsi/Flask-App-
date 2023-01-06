from flask import Flask, render_template, request

from svm_func import train_DT, predict_DT

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from time import time


app = Flask(__name__)
app.url_map.strict_slashes = False

@app.route('/')
def hello_method():
	return render_template('home.html')

@app.route('/predict', methods=['POST']) 
def login_user():

	data = request.form['space']
	print(data)	
	out1= predict_DT(data)
	print(out1[0])
	if(out1==0):
		output = 'G protein coupled receptors'
	if(out1==1):
		output = 'Tyrosine kinase'
	if(out1==2):
		output = 'Tyrosine phosphatase'
	if(out1==3):
		output = 'Synthetase'
	if(out1==4):
		output = 'Synthase'
	if(out1==5):
		output = 'Ion channel'
	if(out1==6):
		output = 'Transcription Factor'
	print(output)
	
	return render_template('result.html', output=output)

@app.route('/profile')
def display():
	return render_template('profile.html')

	
	

if __name__=='__main__':
	global clf
	from waitress import serve 	
	#clf1 = train_DT()	
	print("Done")
	serve(app, host="0.0.0.0", port=8080)

