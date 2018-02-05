import random

import algorithm as ga
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def target_numerical_to_binary(y):
    y = y.shift(-1)
    return y.apply(lambda x: 1 if x > 0.0 else 0)


def create_numerical_direction(df):
    return df['close'] - df['close'].shift(1)


def fill_values(df):
    names = df.columns.get_values()

    for name in names:
        df[name].replace([np.inf, -np.inf], np.nan)
        df[name].fillna(method='ffill', inplace=True)
        df[name].fillna(method='bfill', inplace=True)

    return df


def set_dataframe(df):
    pd.set_option('use_inf_as_na', True)
    df = fill_values(df)
    y_regress = create_numerical_direction(df)
    y = target_numerical_to_binary(y_regress)

    return df, y


def create_dataframe(df, features):
    df, y = set_dataframe(df)
    names = df.columns.get_values()

    for i in range(len(features)):
        if not int(features[i]):
            df = df.drop(labels=names[i], axis=1)

    nb_classes = len(df.columns.get_values())
    x = df.iloc[:, range(nb_classes)].values.astype(np.float)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return df, x_train, x_test, y_train, y_test, nb_classes


def train_and_score(features, df):
    df, x_train, x_test, y_train, y_test, nb_classes = create_dataframe(df, features)
    classifier = SVC(kernel='rbf', random_state=42)
    classifier.fit(x_train, y_train.values.ravel())
    return classifier.score(x_test, y_test)


def train_models(models, df):
    i = 1
    for model in models:
        if '1' not in model.features:
            index = random.randint(0, len(model.features) - 1)
            h = model.features[:index]
            t = model.features[index + 1:]
            model.features = h + str(1 - int(model.features[index])) + t
        print('\tIndividuo ' + str(i) + ' com features ' + model.features)
        model.train(df)
        i += 1


def initial_features(nb_population, feature_size):
    all_features = []
    for _ in range(nb_population):
        features = ''
        for i in range(feature_size):
            features += str(random.randint(0, 1))
        all_features.append(features)
    return all_features


def generate(df, nb_generations=10, nb_population=20):
    df = df.drop(labels='Date', axis=1)
    features = initial_features(nb_population, len(df.columns.get_values()))
    optimizer = ga.GA(features)
    population = optimizer.create_population(nb_population)

    for _ in range(nb_generations):
        print('Geracao ' + str(_+1))
        train_models(population, df)

        if _ != nb_generations - 1:
            population = optimizer.evolve(population)

    population = sorted(population, key=lambda m: m.accuracy, reverse=True)
    return population
