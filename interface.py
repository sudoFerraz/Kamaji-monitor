import model
import auxiliary
import monitor
import strategy_observer
import time
from time import sleep
import numpy as np
import pandas_datareader.data as web
from datetime import datetime
import stockstats
import pandas as pd
from stockstats import StockDataFrame
import datetime as dt
#import matplotlib.pyplot as plt
from pyfiglet import Figlet
from prettytable import PrettyTable
table = PrettyTable()
from datetime import datetime
from datetime import timedelta

while True:
    try:
        monitor.monitor()
    except:
        pass
    try:
        strategy_observer.strategy_observer()
    except:
        pass
    f = Figlet(font='epic')
    print f.renderText('Kamaji')

    os_tools = auxiliary.ostools()
    session = os_tools.db_connection()
    user_handler = auxiliary.user_handler()
    indicator_handler = auxiliary.indicator_handler()
    signal_handler = auxiliary.signal_handler()
    invoice_handler = auxiliary.invoice_handler()


    try:
        df = web.DataReader('BRL=X', 'yahoo')
        df.to_csv('brlusd.csv', mode='w', header=True)
    except:
        pass
    data = StockDataFrame.retype(pd.read_csv('brlusd.csv'))
    macd_histogram = data['macdh']
    last_tendence = macd_histogram[-1]
    if last_tendence > 0:
        last_tendence = 1
    else:
        last_tendence = 0
    tendence = session.query(model.Tendence).first()
    if not tendence:
        if last_tendence > 0:
            new_signal = model.Tendence(flag=1)
        else:
            new_signal = model.Tendence(flag=0)
        session.add(new_signal)
        session.commit()
        session.flush()
    else:
        if last_tendence != tendence.flag:
            if (datetime.now() - tendence.date) > timedelta(days=1):
                tendence.flag = last_tendence
                tendence.date = datetime.now()
                session.commit()
                session.flush()
            else:
                pass
        else:
            pass


   # print "[+][+] Status do mercado no momento [+][+]"
   # print "\n"
   # t = PrettyTable()
   # t.add_column("Close", [close_price_indicator.value])
   # t.add_column("Change 2 dias", [change_2days_ago_indicator.value])
   # print t
   # print "\n"
   # print "[+] Indicadores Ativos [+]"
   # print "\n"
   # print '[+] Ultimo Teto Bollinger Band ' + str(bollinger_up_indicator.value)
   # print '[+] Ultimo Chao Bollinger Band ' + str(bollinger_low_indicator.value)
   # print "[+] Ultimo EMA Bollinger Band " + str(bollinger_indicator.value)
   # print '[+] Ultimo RSI 6 dias ' + str(rsi6_indicator.value)
   # print '[+] Ultimo RSI 12 dias ' +  str(rsi12_indicator.value)
   # print '[+] Ultimo Macd ' + str(macd_indicator.value)
   # print '[+] Ultimo Macd Signal line ' + str(macd_signal_line_indicator.value)
   # print '[+] Ultimo Macd Histogram ' + str(macd_histogram_indicator.value)
   # print "[+] Ultimo SMA 20 dias ", str(close_20_sma[-1])
   # print "[+] Ultimo MSTD 20 dias ", str(close_20_mstd[-1])
   # print "[+] Ultimo EMA 12 dias ", str(close_12_ema[-1])
   # print "[+] Ultimo EMA 26 dias ", str(close_26_ema[-1])
   # print "\n"
   # print "\n"
   # print "********************************************************************"
   # print "\n"
   # print "\n"
   # print "[+] Sinais Ativos [+]"
   # print "\n"
   # table = PrettyTable(["Signal name", "Active", "Relevancia", "Fora do padrao"])
   # table.align["Signal name"] = "1"
   # table.padding_width = 1
   # if bollinger_up_signal:
   #     table.add_row(["Bollinger Band UB", "True", bollinger_up_signal.accuracy, bollinger_up_standardized.value])
   # else:
   #     table.add_row(["Bollinger Band UB", "False", "0", "-"])
   # if bollinger_low_signal:
   #     table.add_row(["Bollinger Band LB", "True", bollinger_low_signal.accuracy, bollinger_low_standardized.value])
   # else:
   #     table.add_row(["Bollinger Band LB", "False", "0", "-"])
   # if macd_histogram_signal:
   #     table.add_row(["MACD Histogram", "True", macd_histogram_signal.accuracy, macd_histogram_standardized.value])
   # else:
   #     table.add_row(["MACD Histogram", "False", "0", "-"])
   # if macd_signal:
   #     table.add_row(["MACD Cross", "True", macd_signal.accuracy, macd_standardized.value])
   # else:
   #     table.add_row(["MACD Cross", "False", "0", "-"])
   # if rsi6_signal:
   #     table.add_row(["Relative Strength Index", "True", rsi6_signal.accuracy, rsi6_standardized.value])
   # else:
   #     table.add_row(["Relative Strength Index", "False", "0", "-"])
#
#    if bollinger_low_signal and macd_histogram_signal and macd_signal and rsi6_signal:
#        buy = Figlet(font='contessa')
#        print buy.renderText('TENDENCIA DE SUBIDA CONSERVADORA')
#    elif bollinger_up_signal:
#        buy = Figlet(font='contessa')
#        print buy.renderText('TENDENCIA DE CORRECAO PARA BAIXO')
#    elif macd_histogram_signal and macd_signal and rsi6_signal:
#        buy = Figlet(font='contessa')
#        print buy.renderText('TENDENCIA DE SUBIDA ARRISCADA')
#    else:
#        hold = Figlet(font='mini')
#        print hold.renderText('TENDENCIA DE RISCO RELATIVO ALTO')
#    print "\n"
#    print table
    sleep(60)
