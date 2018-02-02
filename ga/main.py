import methods
import random


def print_acc(pop):
    f = open('acc_x_fratures.txt', 'a')
    for p in pop:
        print(str(p.accuracy) + ' com features ' + p.features)
        f.write(str(p.accuracy) + '\t\t' + p.features + '\n')
    f.close()


if __name__ == '__main__':
    initial_features = ''
    for i in range(69):
        initial_features += str(random.randint(0, 1))

    population = methods.generate(initial_features, 20, 40)
    print_acc(population[:5])
