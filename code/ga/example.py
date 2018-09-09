#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import genetic_algorithm as ga
import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':
    pd.set_option('use_inf_as_na', True)
    df = pd.read_csv('../../datasets/USDBRL/all_normalized.csv')
    df = df.drop('Date', axis=1)

    t_df = []
    invoice_df = pd.read_csv('../../invoices_forecast.csv')
#    invoice_df1 = pd.read_csv('../../invoices_forecast.csv')
#    invoice_df2 = pd.read_csv('../../invoices_forecast.csv')


    label_df = pd.read_csv('../../label_forecast.csv')
    t_df.append(invoice_df)
#    t_df.append(invoice_df1)
#    t_df.append(invoice_df2)
    t_df.append(label_df)
    y = df['close'] - df['close'].shift(-15)
    # y = y.shift(-1)

    y_regress = y
    y_regress = y_regress.fillna(method='ffill')
    y_regress = np.array(y_regress)
    y_regress = y_regress.reshape(-1, 1)

    y = y.apply(lambda x: 1 if x > 0.0 else 0)

    '''
        Chamada para utilizar o ga com diferentes intervalos de dias.
        Para utilizar a função é necessário passar um CSV de dados, um CSV com as infos.
        Sendo o de dados, os dados para treinar o modelo.

        Esse parametro pode ser tanto um dataframe do pandas, ou uma lista de dataframes, sendo passados no tipo
        primitivo list. => []

        O CSV com as infos são para passar as informações de qual modelo e qual o intervalo de dias usar, e tais
        colunas DEVEM ter o nome de 'Model' e 'Interval', principalmente o 'Interval', para saber o intervalo de dias a
        ser utilizado. O modelo, caso nao encontrado irá cair para o modelo padrão que é o SVM.
        As siglas do modelo no CSV deve seguir a seguinte nomeclatura:
            'svm' -> Modelo SVM
            'crf' -> Random Forrest Classifier
            'nn' -> Para rede neural
            'dtc' -> Decision Tree Classifier

        Valores opcionais são nb_population e nb_generations para serem utilizados no ga.

        O método nao vai retornar nada, ele vai dar um append no CSV de info, com as colunas 'accuracy' e 'predict',
        ao fim do método é escrito todos valores no arquivo 'accuracy_and_predict.csv', junto com as informações que
        foram passadas.
    '''

    ga.calc_with_interval(data_csv=df, info_csv=t_df, nb_population=20, nb_generations=10)

    '''
        Chamando initialize e passando o dataframe, APOS O TRATAMENTO QUE QUISER REALIZAR, ou seja,
        após normalização, todos os valores são númericos, etc.

        y é o target para o modelo.
        Se for classificacao, deixar o target em classes, notei agora que por enquanto tá funcionando só quando é
            binário, vou arrumar pra poder suportar mais classes depois


        O retorno do método é a populacao que foi gerada, com todos os individuos, e um dicionario que contem a acuracia
        no decorrer da evolucao.


        A assinatura do método é a seguinte
        initialize(df, y, nb_generations=10, nb_population=20, model='svm', accuracy=0.6)

        Então passar o datafram e o target sao parametros obrigatorios.
        nb_generation -> numeros de geracoes
        nb_population -> numero de individuos em cada populacao
        model -> qual modelo usar, podend ser ['svm', 'crf', 'nn', 'dtc'] que são svm classificacao,
        random forrest classificacao, rede neural e decision tree classifier, nessa ordem.

        O ultimo parametro é a accuracia para ficar no loop, cuidado pra nao colocar um valor que nao da pra alcançar.
    '''
    #    population, accuracies = ga.initialize(df, y, model='svm')

    '''
        Para acessar a média da acurácia na geracao 3:
    '''
    #    print('Media de acuracia na 3 geracao:')
    #    print(np.mean(accuracies[3]))

    '''
        Selecionando os 5 melhores da populacao
    '''
    #    print('\n\nMelhores 5 acuracias encontradas')
    #    first_five = population[:5]

    '''
        Imprimindo a acuracia dos 5 primeiros
    '''
    #    for individual in first_five:
    #        print(individual.accuracy)

    '''
        Selecionando o melhor modelo, já treinado, e pronto para utilizar. Também pode utilizar para retirar as métricas
        que precisar.
    '''
#    best_model = first_five[0].model
#    print(best_model)
#    print(first_five[0].features)
