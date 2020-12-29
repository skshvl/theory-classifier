#Goal of this .py file: load training data. the X and y data can later be accessed 

import sklearn
import numpy

from sklearn.datasets import load_files

from collections import defaultdict

#training data should be category folders with text files in it



#list of categories to train on. Trains on all if == None
#CATEGORIES = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']

def load_data(directory, categories = None):
    """Load a  directory and generate a scikit bunch"""
    return load_files(directory,
                                            categories = categories,
                                            shuffle = True,
                                            random_state = 42)

#data = load_data(TRAINING_DIRECTORY) #create scikit bunch for the data, assuming cateogry folders
#X_train = data.data #the actual texts as a list
#y_train = data.target #the "target" id's as a list
