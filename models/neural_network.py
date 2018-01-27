#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)


def neural_network_classify(x, y):
    logging.info('\t[+] Splitting 80% to train and 20% to test.')
    print('\t[+] Splitting 80% to train and 20% to test.')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    
    logging.info('\t[+] Creating Neural Network topology')
    print('\t[+] Creating Neural Network topology')
    classifier = Sequential()
    classifier.add(Dense(input_dim=len(x_train[0, :]), units=len(x_train[0, :]), activation='relu',
                         kernel_initializer='uniform'))
    classifier.add(Dense(units=2 * len(x_train[0, :]), activation='relu', kernel_initializer='uniform'))
    classifier.add(Dense(units=2 * len(x_train[0, :]), activation='relu', kernel_initializer='uniform'))
    classifier.add(Dense(units=len(x_train[0, :]), activation='relu', kernel_initializer='uniform'))
    classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))

    logging.info('\t[+] Compiling Neural Network with 3 hidden layers.')
    print('\t[+] Compiling Neural Network with 3 hidden layers.')
    optimizer = RMSprop(lr=0.00075)
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    logging.info('\t[+] Training Neural Network.')
    print('\t[+] Training Neural Network.')
    classifier.fit(x_train, y_train.values.ravel(), batch_size=16, epochs=30, verbose=0)
    
    logging.info('\t[+] Predicting with Neural Network.')
    print('\t[+] Predicting with Neural Network.')
    y_pred = classifier.predict(x_test, batch_size=16)
    
    y_pred = (y_pred > 0.5)
    
    logging.info('\t[+] Creating Neural Network confusion matrix.')
    print('\t[+] Creating Neural Network confusion matrix.')
    return confusion_matrix(y_test, y_pred)
