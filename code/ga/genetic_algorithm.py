import methods
import pandas as pd


def initialize(df, y, nb_generations=10, nb_population=20, model='svm', accuracy=0.6):
    return methods.generate(df, y, nb_generations, nb_population, model, accuracy)


def calc_with_interval(data_csv, info_csv, nb_generations=10, nb_population=20):
    if type(info_csv) == list:
        name_list = list(range(len(info_csv)))
        for name in name_list:
            info_csv[name].name = str(name)
            methods.begin(data_csv=data_csv, df=info_csv[name], nb_generations=nb_generations,
                          nb_population=nb_population)
    elif type(info_csv) == pd.core.frame.DataFrame:
        info_csv.name = 'generated'
        methods.begin(data_csv=data_csv, df=info_csv, nb_generations=nb_generations, nb_population=nb_population)

    else:
        print('Could not determine \'info_csv\' type. Between \'list\' or \'pd.core.frame.DataFrame\'')
