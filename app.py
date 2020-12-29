# import the Flask class from the flask module
from flask import Flask, render_template, request

#load classifier model
from joblib import load
import pickle

import os

#load the classifier
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
classifier_path = os.path.join(THIS_FOLDER, 'classifier.joblib')
CLASSIFIER = load(classifier_path) #import classifier


#load author word counts as dictionary
wordcount_path = os.path.join(THIS_FOLDER, 'word-counts.pickle')
with open(wordcount_path, 'rb') as handle:
    word_counts_imported = pickle.load(handle) #import this [word][author] count dictionary
    
#pickle load sklearn bunch (for data labeling)
data_path = os.path.join(THIS_FOLDER, 'data.pickle')
data = pickle.load(open(data_path, 'rb'))


#load word specificity score for each author
from theorizer import author_word_scores_text, word_score
top_words_each_author = author_word_scores_text(word_score(word_counts_imported))


#load author descriptions as dictionary
from website_methods import csv_to_dict
descriptions = csv_to_dict("author-descriptions.csv")

#load theorizer method
from theorizer import theorize_text

#load method to score new text's word specificity
from theorizer import new_text_top_words



# create the application object
app = Flask(__name__)

# use decorators to link the function to a url
@app.route('/', methods=['GET']) #meaning we "get" something from server
def home():
	"""displays the home page"""
	return render_template('index.html', fieldtext = "Paste text here...") # render a template

#about page
@app.route('/about', methods =['GET'])
def about():
    return render_template('about.html', top_words = top_words_each_author)

#analysis button
@app.route('/conclusion', methods=['POST']) #meaning we UPLOAD to server (I think...)
def analyze():
	"""Route where we send input"""
	
	#we get the relevant form input through request.form
	
	input_text = request.form['Text']
	fieldtext = input_text
	scores = theorize_text(input_text, CLASSIFIER, data=data, dict_result=True)
	ranking = sorted(scores, key=scores.get, reverse=True)
	topguess = ranking[0]
	secondguess = ranking[1]
	
	#render a template with the given parameters
	return render_template('index.html',
			calculation_success = True, 
			topguess=topguess,
			secondguess = secondguess,
			fieldtext = fieldtext,
            topguessdescription = descriptions.get(topguess, "No description found"),
            topwords_newtext = new_text_top_words(input_text, word_counts_imported))  

# start the server with the 'run()' method
if __name__ == '__main__':
    app.run(debug=True) #debug makes sure it keeps updating with changes