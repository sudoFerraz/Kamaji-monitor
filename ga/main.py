import methods
import pandas as pd


def print_acc(pop):
    f = open('acc_x_fratures.txt', 'a')
    f.write('\n\nNova execucao:\n')
    for p in pop:
        print(str(p.accuracy) + ' com features ' + p.features)
        f.write(str(p.accuracy) + '\t\t' + p.features + '\n')
    f.close()


if __name__ == '__main__':
    df = pd.read_csv('../datasets/USDBRL/all_normalized.csv')
    population = methods.generate(df, 50, 100)
    print_acc(population[:10])
