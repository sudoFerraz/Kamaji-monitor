import model
import auxiliary
import time
import numpy as np
import pandas_datareader.data as web
from datetime import datetime
import stockstats
import pandas as pd
from stockstats import StockDataFrame
import datetime as dt
import matplotlib.pyplot as plt
from pyfiglet import Figlet
from prettytable import PrettyTable
table = PrettyTable()

os_tools = auxiliary.ostools()
session = os_tools.db_connection()
user_handler = auxiliary.user_handler()
indicator_handler = auxiliary.indicator_handler()
signal_handler = auxiliary.signal_handler()
invoice_handler = auxiliary.invoice_handler()

bollinger_up_indicator = indicator_handler.get_indicator_by_name(session, 'bollinger_up')
bollinger_up_mean = indicator_handler.get_indicator_by_name(session, 'bollinger_up_mean')
bollinger_up_std = indicator_handler.get_indicator_by_name(session, 'bollinger_up_std')
bollinger_up_standardized = indicator_handler.get_indicator_by_name(session, 'bollinger_up_standardized')
bollinger_low_indicator = indicator_handler.get_indicator_by_name(session, 'bollinger_low')
bollinger_low_mean = indicator_handler.get_indicator_by_name(session, 'bollinger_low_mean')
bollinger_low_std = indicator_handler.get_indicator_by_name(session, 'bollinger_low_std')
bollinger_low_standardized = indicator_handler.get_indicator_by_name(session, 'bollinger_low_standardized')
bollinger_indicator = indicator_handler.get_indicator_by_name(session, 'bollinger')
bollinger_mean = indicator_handler.get_indicator_by_name(session, 'bollinger_mean')
bollinger_std = indicator_handler.get_indicator_by_name(session, 'bollinger_std')
bollinger_standardized = indicator_handler.get_indicator_by_name(session, 'bollinger_standardized')
close_price_indicator = indicator_handler.get_indicator_by_name(session, 'close_price')
close_price_mean = indicator_handler.get_indicator_by_name(session, 'close_mean')
close_price_std = indicator_handler.get_indicator_by_name(session, 'close_std')
close_price_standardized = indicator_handler.get_indicator_by_name(session, 'close_standardized')
rsi6_indicator = indicator_handler.get_indicator_by_name(session, 'rsi6')
rsi6_mean = indicator_handler.get_indicator_by_name(session, 'rsi6_mean')
rsi6_std = indicator_handler.get_indicator_by_name(session, 'rsi6_std')
rsi6_standardized = indicator_handler.get_indicator_by_name(session, 'rsi6_standardized')
rsi12_indicator = indicator_handler.get_indicator_by_name(session, 'rsi12')
rsi12_mean = indicator_handler.get_indicator_by_name(session, 'rsi12_mean')
rsi12_std = indicator_handler.get_indicator_by_name(session, 'rsi12_std')
rsi12_standardized = indicator_handler.get_indicator_by_name(session, 'rsi12_standardized')
macd_indicator = indicator_handler.get_indicator_by_name(session, 'macd')
macd_mean = indicator_handler.get_indicator_by_name(session, 'macd_mean')
macd_std = indicator_handler.get_indicator_by_name(session, 'macd_std')
macd_standardized = indicator_handler.get_indicator_by_name(session, 'macd_standardized')
macd_histogram_indicator = indicator_handler.get_indicator_by_name(session, 'macd_histogram')
macd_histogram_mean = indicator_handler.get_indicator_by_name(session, 'macd_histogram_mean')
macd_histogram_std = indicator_handler.get_indicator_by_name(session, 'macd_histogram_std')
macd_histogram_standardized = indicator_handler.get_indicator_by_name(session, 'macd_histogram_standardized')
macd_signal_line_indicator = indicator_handler.get_indicator_by_name(session, 'macd_signal_line')
macd_signal_line_mean = indicator_handler.get_indicator_by_name(session, 'macd_signal_line_mean')
macd_signal_line_std = indicator_handler.get_indicator_by_name(session, 'macd_signal_line_std')
macd_signal_line_standardized = indicator_handler.get_indicator_by_name(session, 'macd_signal_line_standardized')
change_2days_ago_indicator = indicator_handler.get_indicator_by_name(session, 'change_2days_ago')
change_2days_ago_mean = indicator_handler.get_indicator_by_name(session, 'change_2days_ago_mean')
change_2days_ago_std = indicator_handler.get_indicator_by_name(session, 'change_2days_ago_std')


