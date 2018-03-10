import methods


def initialize(df, y, nb_generations=10, nb_population=20, model='svm', accuracy=0.6):
    return methods.generate(df, y, nb_generations, nb_population, model, accuracy)


def calc_with_interval(data_csv, info_csv, nb_generations=10, nb_population=20):
    methods.begin(data_csv=data_csv, df=info_csv, nb_generations=nb_generations, nb_population=nb_population)
