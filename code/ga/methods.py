import random

import algorithm as ga
import glob
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import trange

pd.set_option('use_inf_as_na', True)
pd.options.mode.chained_assignment = None


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
    x = df.iloc[:, :]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return df, x_train, x_test, y_train, y_test, nb_classes


def predict_values(df, features):
    names = df.columns.get_values()

    for i in range(len(features)):
        if not int(features[i]):
            df = df.drop(labels=names[i], axis=1)

    return df.tail(1)


def train_and_score(features, df, y, model_name, configuration):
    df, x_train, x_test, y_train, y_test, nb_classes = create_dataframe(df, features, y)
    if model_name == 'dtc':
        model = DecisionTreeClassifier(criterion=configuration['DTC']['Criterion'],
                                       random_state=int(configuration['DTC']['Seed']))
        model.fit(x_train, y_train)
        acc = model.score(x_test, y_test)
    elif model_name == 'crf':
        model = RandomForestClassifier(criterion=configuration['CRF']['Criterion'],
                                       random_state=int(configuration['CRF']['Seed']))
        model.fit(x_test, y_test)
        acc = model.score(x_test, y_test)
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
        y_pred = model.predict(x_test, batch_size=16)
        y_pred = (y_pred > 0.5)
        cm = confusion_matrix(y_test, y_pred)
        acc = (cm[0][0] + cm[1][1]) / len(y_pred)
    else:
        model = SVC(C=int(configuration['SVM']['C']),
                    kernel=configuration['SVM']['Kernel'],
                    degree=int(configuration['SVM']['Degree']),
                    gamma=configuration['SVM']['Gamma'],
                    coef0=float(configuration['SVM']['Coef0']),
                    probability=bool(configuration['SVM']['Probability']),
                    shrinking=bool(configuration['SVM']['Shrinking']),
                    tol=float(configuration['SVM']['Tol']),
                    decision_function_shape=configuration['SVM']['Decision'],
                    random_state=int(configuration['SVM']['Seed']),
                    )
        model.fit(x_train, y_train.values.ravel())
        acc = model.score(x_test, y_test)

    return acc, model


def train_models(models, df, y):
    i = 1
    for model in models:
        if '1' not in model.features:
            index = random.randint(0, len(model.features) - 1)
            h = model.features[:index]
            t = model.features[index + 1:]
            model.features = h + str(1 - int(model.features[index])) + t
#        print('\tIndividuo ' + str(i) + ' com features ' + model.features)
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
        accuracies.append(individual.accuracy)

    dict[generation] = accuracies
    return dict


def use_greater_accuracy(df):
    for i in range(len(df)):
        pivot = df.iloc[i]
        for j in range(len(df)):
            to_verify = df.iloc[j]
            if i != j:
                if to_verify['Interval'] == pivot['Interval'] and to_verify['Model'] == pivot['Model']:
                    if to_verify['accuracy'] > pivot['accuracy']:
                        df.at[int(i), 'predict'] = to_verify['predict']
                    else:
                        df.at[int(j), 'predict'] = pivot['predict']

    return df


def begin(data_csv, df, nb_generations=10, nb_population=20, configuration=None):
    df['accuracy'] = pd.Series()
    df['predict'] = pd.Series()
    df['sugestao'] = pd.Series('null', index=range(len(df)))

    for i in trange(len(df)):
        items = df.iloc[i]
        interval, model, y = None, None, None

        cp_data_csv = data_csv

        try:
            interval = getattr(items, 'Interval')

            if isinstance(interval, np.integer):
                y = data_csv['close'] - data_csv['close'].shift(-interval)
                y = y.apply(lambda x: 1 if x > 0.0 else 0)

                if interval > 0:
                    cp_data_csv = data_csv[:len(data_csv) - interval]
                    y = y[:len(y) - interval]
                else:
                    cp_data_csv = data_csv[interval:]
                    y = y[interval:]

            else:
                print('Interval is not int type.')

        except Exception:
            print('Column \'Interval\' not found.')

        try:
            model = getattr(items, 'Model')

        except Exception:
            print('Column \'Model\' not found, default model set to SVM.')

        if y is not None:
            if len(cp_data_csv) == len(y):
                population, accuracies = generate(df=cp_data_csv, y=y, nb_generations=nb_generations,
                                                  nb_population=nb_population, model=model, configuration=configuration)

                last_line = predict_values(cp_data_csv, population[0].features)
                last_line = last_line.values.ravel()
                last_line = last_line.reshape(1, -1)
                pred = population[0].model.predict(last_line)
                pred = pred > 0.5

                df.at[i, 'accuracy'] = population[0].accuracy
                df.at[i, 'predict'] = pred

            else:
                print('Data CSV and target with different sizes.')

        else:
            print('Could not find values for needed properties.')


def generate(df, y, nb_generations=10, nb_population=20, model='svm', configuration=None):
    df = verify_columns(df)
    features = initial_features(nb_population, len(df.columns.get_values()))
    optimizer = ga.GA(features=features,
                      configuration=configuration,
                      retain=float(configuration['DEFAULT']['Retain']),
                      random_select=float(configuration['DEFAULT']['RandSelect']),
                      mutation=float(configuration['DEFAULT']['Mutation']))
    population = optimizer.create_population(nb_population, model)
    generations_accuracies = {}
    _ = 0

    while _ < nb_generations:
        _ += 1
        train_models(population, df, y)

        generations_accuracies = generation_score(generations_accuracies, population, _)
        if _ != nb_generations - 1:
            population = optimizer.evolve(population)

        population = sorted(population, key=lambda m: m.accuracy, reverse=True)

    return population, generations_accuracies


def verify_past_predictions(answer, interval=1):
    recent_predictions_names = glob.glob('./[0-9].csv')

    info = list()

    for name in recent_predictions_names:
        right = 0
        total = 0
        predicted_csv = pd.read_csv(name)

        for index, row in predicted_csv.iterrows():
            if row['Interval'] == interval:
                total += 1

                if row['predict'] == answer:
                    right += 1

        info.append({'csv_name': name,
                     'total_predictions': total,
                     'right_answers': right}
                    )

    return info
