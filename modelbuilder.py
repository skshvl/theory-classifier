#Building a classifier and saving it through joblib.
## Building a scikit classifier


# EXECUTE THIS FILE TO TRAIN THE MODEL AND SAVE IT TO THE CLASSIFIER.JOBLIB FILE


##We build a classifier on a training directory in which folder names are authors, and they contain .txt files with the texts.
##
##We use sklearn's Pipeline method to transform the training data, first vectorizing, then applying TFIDF, then finally the SGD Classifier.
##
##We have the option to enable or disable a feature that selects only the top k features (in this case the top-scoring by TFIDF)

#Finally, we EXPORT the model into a joblib file that we can then load elsewhere without having to train it again.


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.linear_model import SGDClassifier 
from sklearn.feature_selection import SelectKBest

from joblib import dump
import pickle

import os
from collections import defaultdict, Counter
from nltk import ngrams

import string


import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIRECTORY = os.path.join(THIS_FOLDER, 'theory-texts')

def build_classifier(X_train, y_train):
    """Builds classifier specified in text"""

    text_clf = Pipeline([
    ('vectorizer', CountVectorizer(decode_error='ignore', ngram_range = (1,3))), #this part turns the whole corpus into a vector of documents and frequencies
    ('tfidf', TfidfTransformer()), #this one weighs that two-dimensional vector to ascribe greater importance to rarely used words, etc. TF-IDF method.
    #('selector', SelectKBest(k=10000)),
    ('classifier', SGDClassifier())
                            ])
    
    text_clf.fit(X_train, y_train)
    
    return text_clf



def ngram_tokenize(s, rng=(1,1)):
    """Turn text into a lowercase list of ngrams with no punctuation"""
    simpletext = s.translate(str.maketrans('', '', string.punctuation)).lower().split()
    
    ngram_list = []
    
    for N in range(rng[0], rng[1]+1): #iterate thru Ngram n values
        grams = ngrams(simpletext, N) #create ngrams (each ngram is a list of words)
        for gram in grams: #convert each ngram into string and add to ngram list
            ngram_list.append(' '.join(gram))
    
    return ngram_list
    

def listdir_nothidden(directory):
    """List files/folders in a directory EXCLUDING hidden files, whose names start with a dot"""
    files = os.listdir(directory)
    return [filename for filename in files if not filename.startswith(".")]


def clean_words(dictionary, word_list):
    """Set a list of words' counts to 0"""
    for word in word_list:
        try:
            for author in dictionary[word].keys():
                dictionary[word][author]=0
        except:
            print(word, "not found in dictionary")
            
    return dictionary

def word_counts(directory):
    """Extract word count based on assumption that subdirectory name is author name"""
    #make a list of the folders (ignoring hidden files!)
    authorfolders = listdir_nothidden(directory)

    word_count_dict = defaultdict(Counter)

    for authorname in authorfolders:
        for textfile in listdir_nothidden(directory+"/"+authorname):
            print(listdir_nothidden(directory+"/"+authorname))
            f = open(directory+"/"+authorname+"/"+textfile, errors = "ignore", encoding = "utf-8") #open the text file
            textstring = f.read()

            #loop through text file
            for word in ngram_tokenize(textstring, rng=(1,1)): #create ngrams with given range of n
                word_count_dict[word][authorname] += 1 #increment the word's entry for that author name

            f.close() #close the file now that we're done with it       
    
    return word_count_dict


#1. DICTIONARY OF AUTHOR COUNTS

#create the dictionary of word counts per author
    
try:
    word_count_dict = word_counts(TRAINING_DIRECTORY)
    
    #clean the dictionary of SPECIFIC WORDS we know we don't want to be in there
    words_to_remove = ['ect', 'bj', 'eology', 'subj', 'obj', 'subli']
    word_count_dict = clean_words(word_count_dict, words_to_remove)
    
    #pickle the dictionary, storing data (serialize)
    with open('word-counts.pickle', 'wb') as handle:
        pickle.dump(word_count_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
except:
    print("Training data could not be processed. Dictionary of word counts will be pre-loaded")
    
    
#2) BUILD AND DUMP TEXT DATA as SKLEARN BUNCH

from training_loader import load_data


try:
    data = load_data(TRAINING_DIRECTORY) #create scikit bunch for the data, assuming cateogry folders
    X_train = data.data #the actual texts as a list
    y_train = data.target #the "target" id's as a list

    pickle.dump(data, open("data.pickle", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
except:
    print("No training data loaded. Loading pre-existing classifier instead.")
    data_path = os.path.join(THIS_FOLDER, 'data.pickle')
    data = pickle.load(open(data_path, 'rb'))
    X_train = data.data #the actual texts as a list
    y_train = data.target #the "target" id's as a list

#3. BUILD AND DUMP CLASSIFIER

#create classifier
text_clf = build_classifier(X_train, y_train)

#saving the training model to disk!
dump(text_clf, "classifier.joblib", compress=True)



