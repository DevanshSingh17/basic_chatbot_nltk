# Libraries needed for NLP
import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

# Libraries needed for Tensorflow processing
import tensorflow as tf
import numpy as np
import random
import json
#load the intents.json file from your local device

# import our chat-bot intents file
with open('intents.json') as json_data:
    intents = json.load(json_data)
intents
words = []
classes = []
documents = []
ignore = ['?']
# loop through each sentence in the intent's patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each and every word in the sentence
        w = nltk.word_tokenize(pattern)
        # add word to the words list
        words.extend(w)
        # add word(s) to documents
        documents.append((w, intent['tag']))
        # add tags to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
# Perform stemming and lower each word as well as remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore]
words = sorted(list(set(words)))

# remove duplicate classes
classes = sorted(list(set(classes)))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words", words)
# create training data
training = []
output = []
# create an empty array for output
output_empty = [0] * len(classes)

# create training set, bag of words for each sentence
for doc in documents:
    # initialize bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stemming each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is '1' for current tag and '0' for rest of other tags
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# shuffling features and turning it into np.array
random.shuffle(training)
training = np.array(training)

# creating training lists
train_x = list(training[:,0])
train_y = list(training[:,1])
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(10,input_shape=(len(train_x[0]),)))
model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))
model.compile(tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy']) 
model.fit(np.array(train_x), np.array(train_y), epochs=1000, batch_size=8, verbose=1)
model.save("model.pkl")
import pickle
pickle.dump( {'words':words, 'classes':classes}, open( "training_data", "wb" ) )
from keras.models import load_model
model = load_model("model.pkl")
# restoring all the data structures
data = pickle.load( open( "training_data", "rb" ) )
words = data['words']
classes = data['classes']
with open('intents.json') as json_data:
    intents = json.load(json_data)
def clean_up_sentence(sentence):
    # tokenizing the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stemming each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# returning bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words):
    # tokenizing the pattern
    sentence_words = clean_up_sentence(sentence)
    # generating bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
    bag=np.array(bag)
    return(bag)
ERROR_THRESHOLD = 0.30
def classify(sentence):
    # generate probabilities from the model
    bag = bow(sentence, words)
    results = model.predict(np.array([bag]))
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results[0]) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

def response(sentence):
    results = classify(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # a random response from the intent
                    return print(random.choice(i['responses']))

            results.pop(0)
