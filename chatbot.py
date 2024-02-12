import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk

from nltk.stem import WordNetLemmatizer

class Chatbot:
    def __init__(self, intents_file='intents.json', model_file='chatbot_model.h5', words_file='words.pkl', classes_file='classes.pkl', error_threshold=0.25):
        self.intents_file = intents_file
        self.model_file = model_file
        self.words_file = words_file
        self.classes_file = classes_file
        self.error_threshold = error_threshold

        self.lemmatizer = WordNetLemmatizer()
        self.words = pickle.load(open(self.words_file, 'rb'))
        self.classes = pickle.load(open(self.classes_file, 'rb'))
        self.model = tf.keras.models.load_model(self.model_file)
        with open(self.intents_file) as file:
            self.intents = json.loads(file.read())

    def sentence_cleanup(self, sentence):
        s_words = nltk.word_tokenize(sentence)
        s_words = [self.lemmatizer.lemmatize(word) for word in s_words]
        return s_words

    def bag_of_words(self, sentence):
        s_words = self.sentence_cleanup(sentence)
        bag = [0] * len(self.words)
        for word1 in s_words:
            for i, data in enumerate(self.words):
                if word1 == data:
                    bag[i] = 1
        return np.array(bag)

    def class_predict(self, sentence):
        BoW = self.bag_of_words(sentence)
        initial_res = self.model.predict(np.array([BoW]))[0]
        final_result = [[i, r] for i, r in enumerate(initial_res) if r > self.error_threshold]
        final_result.sort(key=lambda x: x[1], reverse=True)
        returnList = []
        for x in final_result:
            returnList.append({'intent': self.classes[x[0]], 'probability': str(x[1])})
        return returnList

    def response(self, res):
        for x in self.intents['intents']:
            if x['tag'] == res:
                result = random.choice(x['responses'])
                break
        return result

    def chat(self):
        while True:
            inp = input()
            if inp == 'exit':
                print('Closing the chatbot.....')
                exit(0)
            intent = self.class_predict(inp)[0]['intent']
            print(self.response(intent))

# Example usage:
chatbot = Chatbot()
chatbot.chat()
