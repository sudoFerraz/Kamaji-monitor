#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import genetic_algorithm as ga
import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':
    pd.set_option('use_inf_as_na', True)
    df = pd.read_csv('../../datasets/USDBRL/all_normalized.csv')
    df = df.drop('Date', axis=1)
    y = df['close'] - df['close'].shift(-1)
    y = y.shift(-1)

    population, accuracies = ga.initialize(df, y)
