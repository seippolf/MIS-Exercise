######################################
#    pip install -U scikit-learn     #
#                                    #
#    pip install -U nltk             #
######################################

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt') #tokenizer
nltk.download('stopwords') #stopwords

clf_svm = svm.SVC(kernel='linear')

FILENAME = 'file.txt'

class Conditions:
    ALLERGY = 'ALLERGY' #Itching and sneezing
    AUTISM = 'AUTISM' #Handwringing and repetitive behavior
    CAVITY = 'CAVITY' #Tooth pain and hole
    STROKE = 'STROKE' #Dizzy and trouble speaking
    VALLEY_FEVER = 'VALLEY FEVER' #Extremely tired and exhausted
    # Five relatively different examples of conditions with 'no' overlapping symptoms.

text = ['I have to sneeze', 'I am itchy', 'He has to itch and scratch', 'Her nose is itchy and she is sneezing',
'I have repetitive behaviors and I was told I handwring', 'He does a lot of handwringing and repeats behaviors', 'I have been called a repetitive person', 'She handwrings and her behaviors are repetitive',
'My tooth hurts all the time and there are holes in some teeth', 'Her tooth has a hole in it and says her tooth hurts', 'I have tooth pain', 'I have a hole in my tooth',
'I feel dizzy and have trouble speaking', 'She can not speak and feels dizzy', 'I find it hard to speak', 'I am dizzy',
'I am extremely exhausted all the time, I just feel constantly tired', 'Why am I so exhausted', 'I noticed she is tired and she said she feels exhausted all the time', 'I feel extremely tired, what is with this exhaustion']
# Examples sentences containing both stopwords and information for finding/attributing patterns.
# Four examples provided for each of the five conditions.

train_conditions = [Conditions.ALLERGY, Conditions.ALLERGY, Conditions.ALLERGY, Conditions.ALLERGY,
Conditions.AUTISM, Conditions.AUTISM, Conditions.AUTISM, Conditions.AUTISM, 
Conditions.CAVITY, Conditions.CAVITY, Conditions.CAVITY, Conditions.CAVITY, 
Conditions.STROKE, Conditions.STROKE, Conditions.STROKE, Conditions.STROKE, 
Conditions.VALLEY_FEVER, Conditions.VALLEY_FEVER, Conditions.VALLEY_FEVER, Conditions.VALLEY_FEVER]

# Trains against text by providing corresponding conditions to each sentence in text

processed_text = []
# This is a list of the sentences from text after stop words were removed and the sentence was stemmed. Used for training

for sentence in text:
    text_tokens = word_tokenize(sentence)
    text_tokens_no_stop = [word for word in text_tokens if not word in stopwords.words()]
    processed_text.append(' '.join([PorterStemmer().stem(i) for i in text_tokens_no_stop]))
# Iterates over text, then tokenizes, removes stop words, stems, and then appends to processed_text.

vectorizer = CountVectorizer(binary=True) # I tried to use specific keywords only once in each example sentence, and turned on Binary to make sure stopwords that slip through skew training less
processed_text_vectors = vectorizer.fit_transform(processed_text)
clf_svm.fit(processed_text_vectors, train_conditions)
# Fits processed_text (now as vectors) to corresponding conditions. Trains on predicting condition based of given words

f = open(FILENAME, 'r').readlines()
f_list = []
for line in f:
    f_list.append(line.rstrip('\n '))
#File I/O. For ease of use with NLP libraries I took the liberty of stripping trailing spaces and newlines (the stemmer may have done this anyway)

for string in f_list:
    f_string = string.lower()
    f_string_tokens = word_tokenize(f_string)
    f_string_tokens_no_stop = [word for word in f_string_tokens if not word in stopwords.words()]
    f_preprocessed = vectorizer.transform([' '.join([PorterStemmer().stem(i) for i in f_string_tokens_no_stop])])
    print(string + ' - ' + clf_svm.predict(f_preprocessed)[0]) 
# Goes through f_list, makes prediction of condition based on provided symptoms. I have it show the line it was reading and then the prediction 

#################################################################################################################################################
#    My rationale for using the bag of words approach was that it was one of the approaches I was more familiar and comfortable with            #   
#    and I felt that going with word vectors tested from a library may have been problematic with medical terminology, as its                   #
#    often going to be repeated in all cases assuming they share even a few words. Therefore I felt a bag of words would be more appropriate.   #
#                                                                                                                                               #
#    Some limitiations to the bag of words approach I took are the lack of semantics. For example 'itching' and 'speaking' are synonymous       #
#    with 'scratching' and 'talking' respectively. The solutions would be to increase the sample size (which could cause other problems) or     #
#    to use a different approach like the word vectors I mentioned before                                                                       #
#################################################################################################################################################

# while(True):
#     user_input = input('Enter symptoms: ').lower() #List of stopwords does not account for case sensitivity, so I converted to lowercase to account for that
#     input_tokens = word_tokenize(user_input)
#     input_tokens_no_stop = [word for word in input_tokens if not word in stopwords.words()]
#     processed_input = ' '
#     # Pre-processes user input ( which i left for demonstration and testing ) before predicting condition.
#     test_condition = vectorizer.transform([processed_input.join([PorterStemmer().stem(i) for i in input_tokens_no_stop])])
#     print(clf_svm.predict(test_condition))
#     # Tests pre-processed user input against its training to predict matching condition.


