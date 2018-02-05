#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)


def random_forest_classify(x, y):
    logging.info('\t[+] Splitting 80% to train and 20% to test.')
    print('\t[+] Splitting 80% to train and 20% to test.')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)

    logging.info('\t[+] Training Random Forest classifier.')
    print('\t[+] Training Random Forest classifier.')
    classifier.fit(x_train, y_train.values.ravel())

    logging.info('\t[+] Predicting with Random Forest.')
    print('\t[+] Predicting with Random Forest.')
    y_pred = classifier.predict(x_test)

    logging.info('\t[+] Creating Random Forest confusion matrix.')
    print('\t[+] Creating Random Forest confusion matrix.')
    return confusion_matrix(y_test, y_pred)
