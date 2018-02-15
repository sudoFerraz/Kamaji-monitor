import random
from operator import attrgetter

import algorithm as ga
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

pd.set_option('use_inf_as_na', True)


def fill_values(df):
    names = df.columns.get_values()

    for name in names:
        df[name].replace([np.inf, -np.inf], np.nan)
        df[name].fillna(method='bfill', inplace=True)
        df[name].fillna(method='ffill', inplace=True)

    return df


def verify_columns(df):
    all_names = df.columns.get_values()
    names = []
    k = 0
    x = df.isnull().any()

    for i in x:
        if i:
            names.append(all_names[k])
        k += 1

    for name in names:
        if df[name].isnull().T.sum() == len(df[name]):
            df = df.drop(labels=[name], axis=1)

    return df


def create_dataframe(df, features, y):
    df = fill_values(df)
    names = df.columns.get_values()

    for i in range(len(features)):
        if not int(features[i]):
            df = df.drop(labels=names[i], axis=1)

    nb_classes = len(df.columns.get_values())
    x = df.iloc[:, range(nb_classes)].values.astype(np.float)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return df, x_train, x_test, y_train, y_test, nb_classes


def count_components(features):
    i = 0
    for component in features:
        if component == '1':
            i += 1

    return i


def train_and_score(features, df, y, model_name):
    df, x_train, x_test, y_train, y_test, nb_classes = create_dataframe(df, features, y)

    if model_name == 'dtc':
        model = DecisionTreeClassifier(criterion='entropy', random_state=42)
        model.fit(x_train, y_train)
        acc_train = model.score(x_train, y_train)
        acc_test = model.score(x_test, y_test)
    elif model_name == 'crf':
        model = RandomForestClassifier(criterion='entropy', random_state=42, oob_score=True)
        model.fit(x_test, y_test)
        acc_train = model.oob_score_
        acc_test = model.score(x_test, y_test)
    elif model_name == 'nn':
        model = Sequential()
        model.add(Dense(input_dim=len(x_train[0, :]), units=len(x_train[0, :]), activation='relu',
                        kernel_initializer='uniform'))
        model.add(Dense(units=2 * len(x_train[0, :]), activation='relu', kernel_initializer='uniform'))
        model.add(Dense(units=2 * len(x_train[0, :]), activation='relu', kernel_initializer='uniform'))
        model.add(Dense(units=len(x_train[0, :]), activation='relu', kernel_initializer='uniform'))
        model.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))
        optimizer = RMSprop(lr=0.00075)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, batch_size=16, epochs=30, verbose=0)
        y_pred_test = model.predict(x_test, batch_size=16)
        y_pred_test = (y_pred_test > 0.5)
        cm = confusion_matrix(y_test, y_pred_test)
        acc_test = (cm[0][0] + cm[1][1]) / len(y_pred_test)
        y_pred_train = model.predict(x_train, batch_size=16)
        y_pred_train = (y_pred_train > 0.5)
        cm = confusion_matrix(y_train, y_pred_train)
        acc_train = (cm[0][0] + cm[1][1]) / len(y_pred_train)
    else:
        model = SVC(kernel='rbf', random_state=42, degree=count_components(features) - 1)
        model.fit(x_train, y_train.values.ravel())
        acc_train = model.score(x_train, y_train)
        acc_test = model.score(x_test, y_test)

    return acc_train, acc_test, model


def train_models(models, df, y):
    i = 1
    for model in models:
        if '1' not in model.features:
            index = random.randint(0, len(model.features) - 1)
            h = model.features[:index]
            t = model.features[index + 1:]
            model.features = h + str(1 - int(model.features[index])) + t
        # TODO remove print
        print(str(i) + ': ' + model.features + '  ' + str(model.accuracy_test))
        model.train(df, y)
        i += 1


def initial_features(nb_population, feature_size):
    all_features = []
    for _ in range(nb_population):
        features = ''
        for i in range(feature_size):
            features += str(random.randint(0, 1))
        all_features.append(features)
    return all_features


def generation_score(dict, population, generation):
    accuracies = []
    for individual in population:
        accuracies.append(individual.accuracy_test)

    dict[generation] = accuracies
    return dict


def generate(df, y, nb_generations=10, nb_population=20, model='svm', accuracy=0.6):
    df = verify_columns(df)
    features = initial_features(nb_population, len(df.columns.get_values()))
    optimizer = ga.GA(features)
    population = optimizer.create_population(nb_population, model)
    generations_accuracies = {}
    acc = 0.0
    _ = 0

    while acc < accuracy or _ < nb_generations:
        print('\n\nNova geracao')
        _ += 1
        train_models(population, df, y)

        generations_accuracies = generation_score(generations_accuracies, population, _)
        if _ != nb_generations - 1:
            population = optimizer.evolve(population)

        population = sorted(population, key=attrgetter('accuracy_test'), reverse=True)
        acc = np.mean(generations_accuracies[_])

    population = sorted(population, key=attrgetter('accuracy_test'), reverse=True)
    return population, generations_accuracies
