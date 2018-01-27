#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import median_absolute_error


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)


def random_forest_regressor(x, y):
    logging.info('\t[+] Splitting 80% to train and 20% to test.')
    print('\t[+] Splitting 80% to train and 20% to test.')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
    
    sc_X = StandardScaler()
    x_train = sc_X.fit_transform(x_train)
    x_test = sc_X.transform(x_test)
    sc_y = StandardScaler()
    y_train = sc_y.fit_transform(y_train)
    y_test = sc_y.fit_transform(y_test)

    logging.info('\t[+] Creating Random Forest regressor.')
    print('\t[+] Creating Random Forest regressor.')
    regressor = RandomForestRegressor(n_estimators = 100, criterion='mse', random_state = 42)
    
    logging.info('\t[+] Training Random Forest regressor.')
    print('\t[+] Training Random Forest regressor.')
    regressor.fit(x_train, y_train.ravel())
    
    y_pred = regressor.predict(x_test)
    
    return median_absolute_error(y_test, y_pred)
