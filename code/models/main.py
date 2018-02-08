#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import models.crandom_forest as crf
import models.neural_network as nn_model
import numpy as np
import pandas as pd
import models.rrandom_forest as rrf
import models.svm as svc
import models.svr as svr
from tqdm import tqdm


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


def model_and_accuracy(df, y, model_name):
    pd.set_option('use_inf_as_na', True)
    df = fill_values(df)
    
    x = df.iloc[:, range(1, 70)].values

    if model_name == 'svr':
        acc, model = svr.svm_regressor(x, y)

    elif model_name == 'nn':
        nn_cm = nn_model.neural_network_classify(x, y)
        acc, model = (nn_cm[0, 0] + nn_cm[1, 1]) / (sum(nn_cm[0]) + sum(nn_cm[1]))

    elif model_name == 'crf':
        crf_cm = crf.random_forest_classify(x, y)
        acc, model = (crf_cm[0, 0] + crf_cm[1, 1])/(sum(crf_cm[0]) + sum(crf_cm[1]))

    elif model_name == 'rrf':
        acc, model = rrf.random_forest_regressor(x, y)

    else:
        svc_cm = svc.svm_classify(x, y)
        acc, model = (svc_cm[0, 0] + svc_cm[1, 1]) / (sum(svc_cm[0]) + sum(svc_cm[1]))

    return model, acc
