import random
import json
import pickle
import numpy as np
import tensorflow as tf
from typing import List, Tuple

import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')


class ChatbotTrainer:
    """
    A class for training a chatbot model.

    Attributes:
        intents_file (str): The path to the JSON file containing the intents.
        epochs (int): The number of epochs to train the model for.
        batch_size (int): The size of the training batch.
        verbose (int): The verbosity level of the training process.
        model_name (str): The name of the file to save the trained model to.
    """

    def __init__(self, intents_file: str, epochs: int = 128, batch_size: int = 16, verbose: int = 1, model_name: str = 'chatbot_model.h5'):
        """
        Initializes a new ChatbotTrainer instance.

        Args:
            intents_file (str): The path to the JSON file containing the intents.
            epochs (int): The number of epochs to train the model for.
            batch_size (int): The size of the training batch.
            verbose (int): The verbosity level of the training process.
            model_name (str): The name of the file to save the trained model to.
        """
        self.intents_file = intents_file
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model_name = model_name

    def train(self) -> None:
        """
        Trains the chatbot model.

        Returns:
            None
        """
        lemmatizer = WordNetLemmatizer()

        with open(self.intents_file) as file:
            intents = json.loads(file.read())

        words: List[str] = []
        classes: List[str] = []
        documents: List[Tuple[List[str], str]] = []
        ignoreLetters = ['?', '!', '.', ',']

        for intent in intents['intents']:
            for pattern in intent['patterns']:
                wordList = nltk.word_tokenize(pattern)
                words.extend(wordList)
                documents.append((wordList, intent['tag']))
                if intent['tag'] not in classes:
                    classes.append(intent['tag'])

        words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
        words = sorted(set(words))
        classes = sorted(set(classes))

        pickle.dump(words, open('words.pkl', 'wb'))
        pickle.dump(classes, open('classes.pkl', 'wb'))

        training = []
        outputEmpty = [0] * len(classes)

        for document in documents:
            bag = []
            wordPatterns = document[0]
            wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
            for word in words:
                bag.append(1) if word in wordPatterns else bag.append(0)

            outputRow = list(outputEmpty)
            outputRow[classes.index(document[1])] = 1
            training.append(bag + outputRow)

        random.shuffle(training)
        training = np.array(training)

        trainX = training[:, :len(words)]
        trainY = training[:, len(words):]

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

        sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        model.fit(trainX, trainY, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        model.save(self.model_name)


# Example usage:
trainer = ChatbotTrainer(intents_file='intents.json', epochs=128, batch_size=16, verbose=1, model_name='chatbot_model.h5')
trainer.train()
