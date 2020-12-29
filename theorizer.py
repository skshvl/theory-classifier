
import string
from collections import defaultdict, Counter
from nltk import ngrams
from math import log

def theorize_text(s, classifier, data, dict_result = True):
	"""Given a string, return decision function as dictionary, where decision assessment is value and author is key
	if dict_result = True, return dict, Otherwise string. data is the sklearn bunch"""

	predictions = classifier.decision_function([s]) #we want to know probabilities! this returns a list of lists of values
	guess_values = defaultdict()
	
	#populate dictionary with decisiion function per author
	for index1, prediction in enumerate(predictions): #loop through predictions (f there are multiple )
		for index2, value in enumerate(prediction): #loop through each guess and the probability
			guess_values[data.target_names[index2]] = value #save prediction to dictionary, getting name of author corresponding to index in prediction    
	if dict_result == True:
		return guess_values #return dictionary of guesses for the given string
	else:
		output = ""
		for author, value in guess_values.items():
			output += author + ": " + str(value)+"\n\n"
	return output

#####################################################
#below methods relate to scoring word uniqueness of a NEW TEXT compared to the training corpus
    
def ngram_tokenize(s, rng=(1,1)):
    """Turn text into a lowercase list of ngrams with no punctuation"""
    simpletext = s.translate(str.maketrans('', '', string.punctuation)).lower().split()
    
    ngram_list = []
    
    for N in range(rng[0], rng[1]+1): #iterate thru Ngram n values
        grams = ngrams(simpletext, N) #create ngrams (each ngram is a list of words)
        for gram in grams: #convert each ngram into string and add to ngram list
            ngram_list.append(' '.join(gram))
    
    return ngram_list

def word_score_formula_new_text(author_word_nr, author_total_words, corpus_word_nr, corpus_total_words):
    """Score words for specificity to a new text -- but only those over a certain number of occurences
    IMPORTANT NOTE: compares this new text's usage to ALL  authors in the training corpus, excluding nobody from original word count dictionary
    this puts it in contrast with the previous version of this method"""
    
    if author_word_nr >= 2.0: #only score those words which have over this many occurences FOR THIS AUTHOR (previous version looked at total corpus occurences)
        result = log(((author_word_nr+1)/(author_total_words+1))/
                     ((corpus_word_nr+1)/(corpus_total_words+1))) #subtract this author's counts from the total
    else:
        result = -9999.0
    
    return result


def new_text_word_score(new_text, corpus_word_count_dict):
    """Given a NEW TEXT, return a dictionary of word idiosyncracy scores for this new text, compared to our training corpus."""

    #generate ngram count dictionary
    newtext_ngrams = ngram_tokenize(new_text)
    new_text_counts = Counter(newtext_ngrams)
    
    #get how many words each author used in corpus, as dictionary of author counts
    total_words_per_author = Counter()
    for word in corpus_word_count_dict.keys(): #iterate through words in our count dict
        for author in corpus_word_count_dict[word].keys(): #iterate through authors
            total_words_per_author[author] += corpus_word_count_dict[word][author] #add that authors count for that word to that author's total word count

    #find out how many words in total in training corpus    
    corpus_word_count = sum(total_words_per_author.values()) 
    
    
    #initialize dictionary. [word] returns idiosyncrasy score for this text's words
    word_scores = defaultdict() 
    
    #total number of new words for the new text
    new_text_totalwords = sum(new_text_counts.values()) 

    for word in new_text_counts.keys(): #iterate through words in our count dictionary for the new text
        corpus_word_nr = sum(corpus_word_count_dict[word].values()) #how many times this word is used in GENERAL IN THE CORPUS

        newtext_word_nr = new_text_counts[word] #how many times the new text uses the word
        
        #we score the idiosyncrasy of this word to this new text
        word_scores[word] = word_score_formula_new_text(newtext_word_nr, new_text_totalwords, corpus_word_nr, corpus_word_count)
        #print(word_scores[author][word])
    return word_scores

def new_text_top_words(new_text, corpus_word_counts):
    """Generate string of a sorted list of most specific words to new text, vs. corpus word counts"""
    newtext_scores = new_text_word_score(new_text, corpus_word_counts)
    sorted_list = sorted(newtext_scores.items(), key=lambda x: x[1], reverse=True)
    
    #now remove the scores
    top_words_list = [item[0] for item in sorted_list]
    return list_to_string(top_words_list[:10])


def list_to_string(items):
    text = ""
    for item in items:
        text += item + ", "
    return text[:-2] #get rid of the last ", "

####################################
    

#below methods score word uniqueness for each author in the corpus, producing a list of top 5 most characteristic words for each author

def word_score_formula(author_word_nr, author_total_words, corpus_word_nr, corpus_total_words):
    """Score words -- but only those over a certain number of occurences
    IMPORTANT NOTE: compares this author's usage to ALL OTHER authors, excluding the one under consideration"""
    
    if corpus_word_nr >= 15.0: #only score those words which have over this many occurences
        result = log(((author_word_nr+1)/(author_total_words+1))/
                     ((corpus_word_nr-author_word_nr+1)/(corpus_total_words-author_total_words+1))) #subtract this author's counts from the total
    else:
        result = -9999.0
    
    return result

def word_score(word_count_dict):
    """Given a word count dictionary, return a dictionary of word idiosyncracy scores like [author][word]"""

    total_words_per_author = Counter()
    for word in word_count_dict.keys(): #iterate through words in our count dict
        for author in word_count_dict[word].keys(): #iterate through authors
            total_words_per_author[author] += word_count_dict[word][author] #add that authors count for that word to that author's total word count

    corpus_word_count = sum(total_words_per_author.values()) #find out how many words in total

    word_scores = defaultdict(defaultdict) #initialize blank two level dictionary of sort [author][word]

    for word in word_count_dict.keys(): #iterate through words in our count dict
        corpus_word_nr = sum(word_count_dict[word].values()) #how many times this word is used in GENERAL
        #print(corpus_word_nr)
        #print(word, corpus_word_nr)
        for author in word_count_dict[word].keys(): #iterate through authors
            author_word_nr = word_count_dict[word][author] #how many times THIS AUTHOR uses the word
            word_scores[author][word] = word_score_formula(author_word_nr, total_words_per_author[author], corpus_word_nr, corpus_word_count)
            #print(word_scores[author][word])
            
    #clean up the results--remove words that didn't meet the minimum occurence threshold:

    #find words to drop
    dropkeys = []
    for author in word_scores.keys():
        for key, value in word_scores[author].items():
            if value == -9999.0:
                dropkeys.append(key)


    #copy score dictionary for author, remove the terms and replace the original dictionary with the new one
    for author in word_scores.keys():
        newdict = word_scores[author]
        for key in dropkeys:
            try:
                newdict.pop(key)
            except:
                continue
        word_scores[author] = newdict #replace word score entry for that author with the newly-created cleaned-up dictionary
        
    return word_scores

def author_word_scores_text(word_scores):
    """Generate a string """
    authors = []
    for author in word_scores.keys():
        ranked_words = sorted(word_scores[author].items(), key=lambda x: x[1], reverse=True)
        wordlist = [item[0] for item in ranked_words] #remove the scores
        authors.append("{}: {}".format(author, list_to_string(wordlist[:8])))        
    return authors

    
#print the top scores for EACH author
#print(author_word_scores_text(word_score(word_count_dict))) #word count dict can be imported from the pickle...