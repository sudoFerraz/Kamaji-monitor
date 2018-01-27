import pandas as pd
import pandas_datareader as web
import stockstats
from stockstats import StockDataFrame

df = web.DataReader('CADCHF=X', 'yahoo')
df.to_csv('brlusd.csv', mode='w', header=True)
data = StockDataFrame.retype(df)
close_price = data['close']
low_bollinger = data['boll_lb']
up_bollinger = data['boll_ub']
rsi6 = data['rsi_6']
rsi12 = data['rsi_12']
macd_signal_line = data['macds']
macd = data['macd']
macd_histogram = data['macdh']
change_1day = data['close_-1_r']
close_20_sma = data['close_20_sma']
close_20_mstd = data['close_20_mstd']
boll = data['boll']
close_12_ema = data['close_12_ema']
close_26_ema = data['close_26_ema']
cr = data['cr']
crma1 = data['cr-ma1']
crma2 = data['cr-ma2']
crma3 = data['cr-ma3']
kdjk = data['kdjk']
kdjd = data['kdjd']
kdjj = data['kdjj']
rsv9 = data['rsv_9']
kdjk9 = data['kdjk_9']
kdjd_9 = data['kdjd_9']
kdjj_9 = data['kdjj_9']
open2sma = data['open_2_sma']
wr10 = data['wr_10']
wr6 = data['wr_6']
cci = data['cci']
cci20 = data['cci_20']
tr = data['tr']
atr = data['atr']
dma = data['dma']
pdi = data['pdi']
mdi = data['mdi']
dx = data['dx']
adx = data['adx']
adxr = data['adxr']
mdm14ema = data['mdm_14_ema']
mdm14 = data['mdm_14']
trix = data['trix']
matrix = data['trix_9_sma']



data.to_csv('all_inticators.csv', mode='w', header=True)
close_price.to_csv('close_price.csv', mode='w', header=True)
low_bollinger.to_csv('low_bolinger.csv', mode='w', header=True)
up_bollinger.to_csv('up_bollinger.csv', mode='w', header=True)
rsi6.to_csv('rsi6.csv', mode='w', header=True)
rsi12.to_csv('rsi12.csv', mode='w', header=True)
macd_signal_line.to_csv('macd_signal_line.csv', mode='w', header=True)
macd.to_csv('macd.csv', mode='w', header=True)
macd_histogram.to_csv('macd_histogram.csv', mode='w', header=True)
change_1day.to_csv('change_1day.csv', mode='w', header=True)
close_20_sma.to_csv('close_20_sma.csv', mode='w', header=True)
close_20_mstd.to_csv('close_20_mstd.csv', mode='w', header=True)
boll.to_csv('boll.csv', mode='w', header=True)
close_12_ema.to_csv('close_12_ema.csv', mode='w', header=True)
close_26_ema.to_csv('close_26_ema.csv', mode='w', header=True)
cr.to_csv('cr.csv', mode='w', header=True)
crma1.to_csv('crma1.csv', mode='w', header=True)
crma2.to_csv('crma2.csv', mode='w', header=True)
crma3.to_csv('crma3.csv', mode='w', header=True)
kdjk.to_csv('kdjk.csv', mode='w', header=True)
kdjd.to_csv('kdjd.csv', mode='w', header=True)
kdjj.to_csv('kdjj.csv', mode='w', header=True)
rsv9.to_csv('rsv9.csv', mode='w', header=True)
kdjk9.to_csv('kdjk9.csv', mode='w', header=True)
kdjd_9.to_csv('kdjd9.csv', mode='w', header=True)
kdjj_9.to_csv('kdjj9.csv', mode='w', header=True)
open2sma.to_csv('open2sma.csv', mode='w', header=True)
wr10.to_csv('wr10.csv', mode='w', header=True)
wr6.to_csv('wr6.csv', mode='w', header=True)
cci.to_csv('cci.csv', mode='w', header=True)
cci20.to_csv('cci20.csv', mode='w', header=True)
tr.to_csv('tr.csv', mode='w', header=True)
atr.to_csv('atr.csv', mode='w', header=True)
dma.to_csv('dma.csv', mode='w', header=True)
pdi.to_csv('pdi.csv', mode='w', header=True)
mdi.to_csv('mdi.csv', mode='w', header=True)
dx.to_csv('dx.csv', mode='w', header=True)
adx.to_csv('adx.csv', mode='w', header=True)
mdm14ema.to_csv('mdm14ema.csv', mode='w', header=True)
mdm14.to_csv('mdm14.csv', mode='w', header=True)
trix.to_csv('trix.csv', mode='w', header=True)
matrix.to_csv('matrix.csv', mode='w', header=True)

#series to df
#macd = pd.DataFrame({'Date':macd.index, 'macd':macd.values})

#for i, row in macd.iterrows():
        #macd.loc[i, 'macd'] = 1
