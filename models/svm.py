#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)


def svm_classify(x, y):
    logging.info('\t[+] Splitting 80% to train and 20% to test.')
    print('\t[+] Splitting 80% to train and 20% to test.')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    classifier = SVC(kernel='rbf', random_state=0)

    logging.info('\t[+] Training SVM classifier.')
    print('\t[+] Training SVM classifier.')
    classifier.fit(x_train, y_train.values.ravel())

    logging.info('\t[+] Predicting with SVM classifier.')
    print('\t[+] Predicting with SVM classifier.')
    y_pred = classifier.predict(x_test)

    logging.info('\t[+] Creating SVM confusion matrix.')
    print('\t[+] Creating SVM confusion matrix.')
    return confusion_matrix(y_test, y_pred)
