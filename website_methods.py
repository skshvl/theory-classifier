
import csv

import os
   
CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))

def csv_to_dict(filename):
    """read csv as dictionary. fist column is key, second column is value"""
    path = os.path.join(CURRENT_FOLDER, filename)
    reader = csv.reader(open(path))
    descriptions = {}
    for row in reader:
        descriptions[row[0]] = row[1]
            
    return descriptions

