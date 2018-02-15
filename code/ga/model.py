import methods as methods


class Model:
    def __init__(self, features, model_name):
        self.accuracy_train = 0.0
        self.accuracy_test = 0.0
        self.components = 0
        self.fitness = 0.0
        self.features = features
        self.model_name = model_name
        self.model = None

    def define_features(self, features):
        self.features = features

    def define_components(self):
        self.components = methods.count_components(self.features)

    def define_fitness(self, value):
        self.fitness = value

    def train(self, df, y):
        if not self.accuracy_train and not self.accuracy_test:
            self.accuracy_train, self.accuracy_test, self.model = methods.train_and_score(self.features, df, y, self.model_name)
            self.define_components()
