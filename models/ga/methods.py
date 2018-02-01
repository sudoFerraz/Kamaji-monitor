from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import math

from sklearn.svm import SVC


def target_numerical_to_binary(y):
    return y['Values'].apply(lambda x: 1 if x > 0.0 else 0)


def create_numerical_direction(df):
    x = np.zeros(len(df), dtype=np.float64)

    if len(df['open']) == len(df['close']):
        for _ in range(len(df['open'])):
            k = (df['high'][_] - df['open'][_]) - (df['close'][_] - df['low'][_])
            v = math.sqrt((df['close'][_] - df['boll_lb'][_]) * (df['close'][_] - df['boll_lb'][_]))
            u = math.sqrt((df['open'][_] - df['boll_ub'][_]) * (df['open'][_] - df['boll_ub'][_]))
            r = (df['rsi_6'][_] + df['rsi_12'][_]) / 200
            x[_] = (r * (df['middle'][_] * k) + (v - u) * df['macd'][_]) / (r + df['macd'][_])

    return pd.DataFrame(data=x, index=range(len(x)), columns=['Values'])


def fill_values(df):
    names = df.columns.get_values()

    for name in names:
        df[name].replace([np.inf, -np.inf], np.nan)
        df[name].fillna(method='ffill', inplace=True)
        df[name].fillna(method='bfill', inplace=True)

    return df


def get_dataframe():
    df = pd.read_csv('../../datasets/USDBRL/all_inticators.csv')
    df = fill_values(df)
    y_regress = create_numerical_direction(df)
    y = target_numerical_to_binary(y_regress)

    return df, y


def create_dataframe(features):
    df, y = get_dataframe()
    names = df.columns.get_values()

    for i in range(len(features)):
        if not int(features[i]):
            df = df.drop(labels=names[i], axis=1)

    nb_classes = len(df.columns.get_values())
    x = df.iloc[:, range(nb_classes)].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return df, x_train, x_test, y_train, y_test, nb_classes


def train_and_score(features):
    df, x_train, x_test, y_train, y_test, nb_classes = create_dataframe(features)
    classifier = SVC(kernel='rbf', random_state=42)
    classifier.fit(x_train, y_train.values.ravel())
    return classifier.score(x_test, y_test)

