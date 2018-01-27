#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import svm as svc
import neural_network as nn_model
import crandom_forest as crf
import rrandom_forest as rrf
import svr as svr


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)


def create_direction(df):
    x = np.zeros(len(df), dtype=np.float64)
    
    if len(df['open']) == len(df['close']):
        for _ in tqdm(range(len(df['open']))):
            x[_] = df['open'][_] - df['close'][_]
            x[_] = 0 if x[_] < 0.0 else 1
    else:
        logging.error('[-] \'open\' and \'close\' columns with different sizes.')
        print('[-] \'open\' and \'close\' columns with different sizes!')
    
    return pd.DataFrame(data=x, index=range(len(x)), columns=['Label'])


def create_numerical_direction(df):
    x = np.zeros(len(df), dtype=np.float64)
    
    if len(df['open']) == len(df['close']):
        for _ in tqdm(range(len(df['open']))):
            a = df['open'][_] - df['high'][_]
            b = df['low'][_] - df['close'][_]
            x[_] = (df['close_20_sma'][_] * (a - b)) * 10
    else:
        logging.error('[-] \'open\' and \'close\' columns with differents sizes.')
        print('[-] \'open\' and \'close\' columns with differents sizes!')
    
    return pd.DataFrame(data=x, index=range(len(x)), columns=['Reg_Value'])


def fill_values(df):
    names = df.columns.get_values()
    
    bar = tqdm(total=len(names))
    for name in names:
        df[name].replace([np.inf, -np.inf], np.nan)
        df[name].fillna(method='ffill', inplace=True)
        df[name].fillna(method='bfill', inplace=True)
        bar.update(1)
    
    bar.close()
    
    return df


if __name__ == '__main__':
    pd.set_option('use_inf_as_na', True)

    logging.info('[+] Loading \'all indicators.csv\' datafile.')
    print('[+] Loading \'all indicators.csv\' datafile.')
    all_indicators = pd.read_csv('../datasets/USDBRL/all_inticators.csv')
    
    logging.info('[+] Filling NaN.')
    print('[+] Filling NaN.')
    all_indicators = fill_values(all_indicators)
    
    logging.info('[+] Creating label from \'open\' and \'close\'.')
    print('\n[+] Creating label from \'open\' and \'close\'.')
    y_classify = create_direction(all_indicators)
    
    logging.info('[+] Creating value from \'open\' and \'close\'.')
    print('\n[+] Creating value from \'open\' and \'close\'.')
    y_regress = create_numerical_direction(all_indicators)
    
    x = all_indicators.iloc[:, range(1, 70)].values
    
    logging.info('[+] Sending data to SVM classifier.')
    print('\n\n[+] Sending data to SVM classifier.')
    svc_cm = svc.svm_classify(x, y_classify)
    print('\t[+] SVC: ' + str((svc_cm[0, 0] + svc_cm[1, 1])/(sum(svc_cm[0]) + sum(svc_cm[1]))))
    
    logging.info('[+] Sending data to SVM regressor.')
    print('\n\n[+] Sending data to SVM regressor.')
    svr_mae = svr.svm_regressor(x, y_regress)
    print('\t[+] SVM regressor mae: ' + str(svr_mae))
    
    logging.info('[+] Sending data to Neural Network classifier.')
    print('\n\n[+] Sending data to Neural Network classifier.')
    nn_cm = nn_model.neural_network_classify(x, y_classify)
    print('\t[+] Neural network: ' + str((nn_cm[0, 0] + nn_cm[1, 1])/(sum(nn_cm[0]) + sum(nn_cm[1]))))
    
    logging.info('[+] Sending data to Random Forest classifier.')
    print('\n\n[+] Sending data to Random Forest classifier.')
    crf_cm = crf.random_forest_classify(x, y_classify)
    print('\t[+] Random forest classify: ' + str((crf_cm[0, 0] + crf_cm[1, 1])/(sum(crf_cm[0]) + sum(crf_cm[1]))))

    logging.info('[+] Sending data to Random Forest regressor.')
    print('\n\n[+] Sending data to Random Forest regressor.')
    rrf_mae = rrf.random_forest_regressor(x, y_regress)
    print('\t[+] Random forest regressor mae: ' + str(rrf_mae))
