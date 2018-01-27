#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from sklearn.metrics import median_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)


def svm_regressor(x, y):
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    x = sc_x.fit_transform(x)
    y = sc_y.fit_transform(y)

    logging.info('\t[+] Splitting 80% to train and 20% to test.')
    print('\t[+] Splitting 80% to train and 20% to test.')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    regressor = SVR(kernel='rbf')

    logging.info('\t[+] Training SVM regressor.')
    print('\t[+] Training SVM regressor.')
    regressor.fit(x_train, y_train.ravel())

    logging.info('\t[+] Predicting with SVM regressor.')
    print('\t[+] Predicting with SVM regressor.')
    y_pred = regressor.predict(x_test)

    logging.info('\t[+] Creating SVM regressor mean absolute error.')
    print('\t[+] Creating SVM regressor mean absolute error.')
    return median_absolute_error(y_test, y_pred)
