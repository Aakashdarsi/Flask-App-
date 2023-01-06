import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from time import time
from sklearn import tree
import pandas as pd
import numpy as np
import Bio
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop
from keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold,cross_val_score,cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1

def getKmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

def train_DT():
	global X_test, y_test
	human = pd.read_table('human_data.txt')
	human['words'] = human.apply(lambda x: getKmers(x['sequence']), axis=1)
	human = human.drop('sequence', axis=1)
	human_texts = list(human['words'])
	for item in range(len(human_texts)):
		human_texts[item] = ' '.join(human_texts[item])
	y_data = human.iloc[:, 0].values
	from sklearn.feature_extraction.text import CountVectorizer
	cv = CountVectorizer(ngram_range=(4,4))
	X = cv.fit_transform(human_texts)	
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size = 0.2, random_state = 42)
	nb_ =MultinomialNB(alpha=0.01)
	nb_.fit(X_train, y_train)
	
	return nb_

def predict_DT(inp):
	#t = time()
	#from sklearn.feature_extraction.text import CountVectorizer
	#cv = CountVectorizer(ngram_range=(4,4))
	#inp = cv.transform(inp)	
	#output = clf1.predict(inp)
	#acc = clf1.predict_proba(inp)
	#print("The running time: ",time()-t)
	with open('input.txt', 'w') as f:
		f.write ("sequence\n")
		f.write(inp)
	human1 = pd.read_table('input.txt')
	human1['words'] = human1.apply(lambda x: getKmers(x['sequence']), axis=1)	
	human_texts1 = list(human1['words'])
	for item in range(len(human_texts1)):
		human_texts1[item] = ' '.join(human_texts1[item])
	global X_test, y_test
	human = pd.read_table('human_data.txt')
	human['words'] = human.apply(lambda x: getKmers(x['sequence']), axis=1)
	human = human.drop('sequence', axis=1)
	human_texts = list(human['words'])
	for item in range(len(human_texts)):
		human_texts[item] = ' '.join(human_texts[item])
	y_data = human.iloc[:, 0].values
	from sklearn.feature_extraction.text import CountVectorizer
	cv = CountVectorizer(ngram_range=(4,4))
	X = cv.fit_transform(human_texts)	
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size = 0.2, random_state = 42)
	nb_ =MultinomialNB(alpha=0.01)
	nb_.fit(X_train, y_train)
	print(nb_.predict(X_test[0:10]))
	comment2 = cv.transform(human_texts1).toarray()
	print("hhhhhhhhhhhhhhhhhhhhhhhh")
	print(comment2)
	output=nb_.predict(comment2)
	return output

