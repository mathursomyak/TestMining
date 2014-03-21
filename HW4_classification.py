__author__ = 'skmathur'
import os
import csv
import numpy as np
from random import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from operator import itemgetter
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import regexp_tokenize
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#------------------------------
#Get Data
#------------------------------
def get_amazon_reviews(filepath='/Users/skmathur/Documents/Analytics/TextMining/HW5/amazon'):
    os.chdir(filepath)
    amazon_reviews = []
    target_labels = []
    for infile in os.listdir(os.path.join(os.getcwd())):
        if infile.endswith('csv'):
            label = infile.split('.')[0]
            target_labels.append(label)

            with open(infile, 'rb') as csvfile:
                amazon_reader = csv.DictReader(csvfile, delimiter=',')
                infile_rows = [{ label: row['review_text'] } for row in amazon_reader]

            for doc in infile_rows:
                amazon_reviews.append(doc)
    return amazon_reviews,target_labels
amazon_reviews, target_labels = get_amazon_reviews()

#------------------------------
# Get Training and Test Set-up
#------------------------------
x = [amazon_reviews[i] for i in range(len(amazon_reviews))]
shuffle(x)
trainset_size = int(round(len(amazon_reviews)*0.75)) # i chose this threshold arbitrarily...
X_train = np.array([''.join(el.values()) for el in x[0:trainset_size]])
y_train = np.array([''.join(el.keys()) for el in x[0:trainset_size]])
y_test = np.array([''.join(el.keys()) for el in x[trainset_size+1:len(amazon_reviews)]])
X_test = np.array([''.join(el.values()) for el in x[trainset_size+1:len(amazon_reviews)]])

'''
Your pipeline should chain the vectorizer with the Multinomial Naive Bayes Classier. Your
 grided search should simultaneously vary  in the classier, the term frequency cuto min df
 and toggle between using a PorterStemmer versus the WordNetLemmatizer in the tf-idf vectorizer.
'''

#------------------------------
#Stemmers for Vectorizer
#------------------------------
stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

matchingexpression = r'([A-Za-z][A-Za-z\+\'\-]+)|[\$\!\&\*]'
def Porter_tokenize(text):
    tokens = regexp_tokenize(text,pattern = matchingexpression)
    stems = stem_tokens(tokens,stemmer=stemmer)
    return stems

lemmatizer = WordNetLemmatizer()
def stem_tokens_lemmatize(tokens):
   stemmed = []
   for item in tokens:
       stemmed.append(lemmatizer.lemmatize(item))
   return stemmed

def Lemmer_tokenize(text):
   # only accept words that begin with an alphabet
   tokenizer = RegexpTokenizer('[A-Za-z]\w+')
   tokens = tokenizer.tokenize(text)
   stems = stem_tokens_lemmatize(tokens)
   return stems


#------------------------------
#Pipeline
#------------------------------
pipeline = Pipeline((
    ('vec', TfidfVectorizer(min_df=0.1, stop_words='english', use_idf=True,strip_accents='unicode', ngram_range=(1,2), norm='l2')),
    ('clf', MultinomialNB(alpha=0.1)),
))

parameters = {
    'vec__tokenizer': [Porter_tokenize,Lemmer_tokenize],
    'vec__min_df': [0, 1, 2],
    'clf__alpha': [0.001,0.01,0.1,1]
    ,'vec__ngram_range':[(1,1),(1,3)] #COMMENT THIS LINE IN ONLY FOR PART D
}

print "Question 1: Part A"
for i in range(len(pipeline.steps)):
    print i,'step: ',pipeline.steps[i]

#------------------------------
# GRID SEARCH
#------------------------------
gs = GridSearchCV(pipeline, parameters, verbose=2, refit=False)
_ = gs.fit(X_train, y_train)

print "Question 1: Part B"
print 'best score', gs.best_score_
print 'best param', gs.best_params_
print 'Comment on the effects of the different parameters on the accuracy of the classifier.'
print 'It seems that the porter stemmer leads to higher accuracy than the Word Lemmatizer in this case. \
    Between 0,1,2 the best min_df is 0. However, between 1 and 2, 2 is better'


fit_ = pipeline.fit(X_train, y_train)

vec_name, vec = pipeline.steps[0]
clf_name, clf = pipeline.steps[1]

feature_names = vec.get_feature_names()
target_names = target_labels
feature_weights = clf.coef_
print 'weights shape: ',feature_weights.shape
#print 'clf.coef_', clf.coef_

print "Question 1: Part C"
predicted = pipeline.predict(X_test)
print(classification_report(y_test, predicted,target_names=target_labels))

print "Confusion Matrix:"
print confusion_matrix(y_test, predicted)
print target_labels

print 'Most Important Features:'
def display_important_features(feature_names, target_names, weights, n_top=30):
    for i, target_name in enumerate(target_names):
        print("Class: " + target_name)
        print("")

        sorted_features_indices = weights[i].argsort()[::-1]

        most_important = sorted_features_indices[:n_top]
        print(", ".join("{0}: {1:.4f}".format(feature_names[j], weights[i, j])
                        for j in most_important))
        print("...")

        least_important = sorted_features_indices[-n_top:]
        print(", ".join("{0}: {1:.4f}".format(feature_names[j], weights[i, j])
                        for j in least_important))
        print("")
display_important_features(feature_names, target_names, feature_weights)

print "Question 1: Part D"
print 'Describe the effect on the accuracy of the classifier by including bi-\
grams in addition to uni-grams in the tf-idf matrix. What is the likely\
reason for such an effect?'
print 'The effect on accuracy is minimal, but it takes WAY longer to run. It is much more computationally expensive.\
        I included bi-grams and tri-grams which I expected would improve the accuracy much more than it did.'