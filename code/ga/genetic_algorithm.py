import methods
import os
import time
from datetime import datetime
import pandas as pd


def initialize(df, y, configuration, nb_generations=10, nb_population=20, model='svm', accuracy=0.6):
    return methods.generate(df, y, nb_generations, nb_population, model, accuracy, configuration)


def save_dataframe(df):
    if not os.path.exists('./dataframes'):
        try:
            os.makedirs('./dataframes')
        except OSError:
            print('[-] Could not create directory to dataframes')

    name = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + '.csv'
    print('[+] Saved file into ./dataframes/' + name)
    df.to_csv(os.path.join('./dataframes/', name))


def calc_with_interval(data_csv, info_csv, nb_generations=10, nb_population=20, configuration=None):
    if type(info_csv) == list:
        name_list = list(range(len(info_csv)))
        for name in name_list:
            info_csv[name].name = str(name)
            methods.begin(data_csv=data_csv, df=info_csv[name], nb_generations=nb_generations,
                          nb_population=nb_population, configuration=configuration, suffix=name)

        for k in range(len(info_csv)):
            pivot_df = info_csv[k]
            for n in range(len(info_csv)):
                verify_df = info_csv[n]
                for i in range(len(pivot_df)):
                    pivot_row = pivot_df.iloc[i]
                    for j in range(len(verify_df)):
                        to_verify = verify_df.iloc[j]
                        if to_verify['Interval'] == pivot_row['Interval'] and \
                                to_verify['Model'] == pivot_row['Model']:

                            if to_verify['accuracy'] > pivot_row['accuracy']:
                                pivot_df.at[i, 'predict'] = to_verify['predict']
                            else:
                                verify_df.at[j, 'predict'] = pivot_row['predict']
        for df in info_csv:
            df.to_csv('./' + df.name + '.csv')
            save_dataframe(df)

    elif type(info_csv) == pd.core.frame.DataFrame:
        info_csv.name = 'generated'
        methods.begin(data_csv=data_csv, df=info_csv, nb_generations=nb_generations, nb_population=nb_population)
        info_csv.to_csv('./' + info_csv.name + '.csv')
        save_dataframe(info_csv)

    else:
        print('Could not determine \'info_csv\' type. Must be \'list\' or \'pd.core.frame.DataFrame\'')
