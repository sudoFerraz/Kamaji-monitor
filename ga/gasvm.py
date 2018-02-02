import methods


class SVM:
    def __init__(self, features):
        self.accuracy = 0.0
        self.features = features

    def define_features(self, features):
        self.features = features

    def train(self):
        if not self.accuracy:
            self.accuracy = methods.train_and_score(self.features)
