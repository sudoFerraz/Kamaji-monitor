import random
from models.ga.gasvm import SVM
from _operator import add
from functools import reduce


class GA:
    def __init__(self, features, retain=0.4, random_select=0.1, mutation=0.2):
        self.features = features
        self.retain = retain
        self.random_select = random_select
        self.mutation = mutation

    def create_population(self, count):
        pop = []
        for _ in range(count):
            model = SVM(self.features)
            pop.append(model)

    @staticmethod
    def fitness(model):
        return model.accuracy

    def grade(self, pop):
        all = reduce(add, (self.fitness(model) for model in pop))
        return all / len(pop)

    def breed(self, individual_a, individual_b):
        pass

    def mutate(self, features):
        index = random.randint(0, len(features))
        features[index] = str(1 - int(features[index]))
        return features

    def evolve(self, pop):
        pass