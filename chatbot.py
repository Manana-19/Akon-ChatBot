import random
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

intents = ''
with open('intents.json') as file:
    intents=json.loads(file.read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

model = tf.keras.models.load_model('chatbot_model.h5')

def sentenceCleanUp(sentence):
    s_words = nltk.word_tokenize(sentence)
    s_words = [lemmatizer.lemmatize(word) for word in s_words]
    
    return s_words

def bagofwords(sentence):
    
    s_words = sentenceCleanUp(sentence)
    bag = [0]*len(words)
   
    for word1 in s_words:
   
        for i, data in enumerate(words):
   
            if word1==data:
                bag[i] = 1
   
   
    return np.array(bag)

def ClassPredict(sentence):

    BoW = bagofwords(sentence)
    initial_res = model.predict(np.array([BoW]))[0]
    ERR_TH = 0.25

    final_result = [[i,r] for i,r in enumerate(initial_res) if r>ERR_TH]
    final_result.sort(key=lambda x: x[1], reverse=True)

    returnList = []

    for x in final_result:
        returnList.append({'intent':classes[x[0]],'probability':str(x[1])})

    return returnList

def response(dictionary_of_intents, res):
    for x in dictionary_of_intents['intents']:
        if x['tag'] == res:
            result=random.choice(x['responses'])
            break
    return result


print('✅')
print("Type 'exit123' in order to exit the application")
print('You\'re good to go with this chatbot')

while True:

    inp = input('=> ')

    if inp == 'exit123': print('Closing the chatbot.....');exit(0)
    # print(ClassPredict(inp))

    intent=ClassPredict(inp)[0]['intent']

    print('Akon: ',response(intents, intent))