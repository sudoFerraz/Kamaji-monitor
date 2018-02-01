from models.ga.methods import train_and_score


class SVM:
    def __init__(self, features):
        self.accuracy = 0.0
        self.features = features

    def train(self):
        if not self.accuracy:
            self.accuracy = train_and_score(self.features)
