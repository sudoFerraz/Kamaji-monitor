#Flask API server

import auxiliary
from flask import Flask, render_template
from flask import jsonify
from flask import request
import json
import pandas as pd
from stockstats import StockDataFrame
import model
from model import Users, Notification, Action, Signal, Raw_data, Indicator
import flask
from flask_sqlalchemy import SQLAlchemy
import sqlalchemy
from flask_admin import Admin
import pandas_datareader as web
from datetime import datetime, timedelta

f = web.DataReader("BRL=X", 'yahoo')



app = Flask(__name__, template_folder='')

app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:postgres@localhost/postgres'
app.config['SECRET_KEY'] = 'postgres'

db = SQLAlchemy(app)
admin = Admin(app)

os_tools = auxiliary.ostools()
session = os_tools.db_connection()
user_handler = auxiliary.user_handler()
strategy_handler = auxiliary.strategy_handler()
data_handler= auxiliary.data_handler()
signal_handler = auxiliary.signal_handler()
contact_handler = auxiliary.contact_handler()
notification_handler = auxiliary.notification_handler()
action_handler = auxiliary.action_handler()
invoice_handler = auxiliary.invoice_handler()
indicator_handler = auxiliary.indicator_handler()
strategy_type_handler = auxiliary.strategy_type_handler()

@app.teardown_request
def app_teardown(response_or_exc):
    session.remove()
    return response_or_exc

#class MyModelView(ModelView):
 #   def __init__(self, model, session, name=None, category=None, endpoint=None,\
  #               url=None, **kwargs):
   #     for k, v in kwargs.iteritems():
    #        setattr(self, k, v)
     #       super(MyModelView, self).__init__(model, session, name=name, category=category, endpoint=endpoint, url=url)

#    def is_accessible(self):
 #       return True

@app.route('/')
def index():
    return "Index"

@app.route('/teste')
def teste():
    return "teste"



@app.route('/checkbox/<int:choice>', methods=['GET'])
def checkbox_markup(choice):
    if request.method == 'GET':
        found = strategy_type_handler.update_strategy_type(session, choice)
        if found:
            return "ok"
        else:
            return "erro"

#Fazer retornar os codigos
@app.route('/login/<string:email>/<string:password>', methods=['GET'])
def login(email, password):
    if request.method == 'GET':
        found = user_handler.verify_user(session, email, password)
        if not found:
            return "erro"
        if found:
            req = {}
            req['name'] = found.name
            req['usertype'] = found.usertype
            req = json.dumps(req)
            return req
"""

#rota de login
@app.route('/login/<string:email>/<string:password>')
def register():
    if request.method == 'GET':
        return True
    elif request.method == 'POST':
        newuser = user_handler.create_user(email, username, password)
        if newuser:
            return True
        else:
            return False
            """

@app.route('/contact/delete/<string:email>')
def delete_contact(email):
    if request.method == 'GET':
        try:
            contact_handler.delete_contact(session, email)
            return "OK"
        except:
            return "ERRO"


type_handler = auxiliary.type_handler()

@app.route('/invoice/delete/<string:invoice_id>')
def delete_invoice(invoice_id):
    invoice_handler.delete_invoice(session, invoice_id)
    return "ok"

@app.route('/invoice/notification/activate/<string:invoice_id>')
def active_invoice_notification(invoice_id):
    return "ok"


@app.route('/contact/register/<string:name>/<string:email>/<string:phone>')
def register_contact(name, email, phone):
    if request.method == 'GET':
        try:
            new_contact = contact_handler.create_contact(session, name, email, phone)
            return "OK"
        except:
            return "ERRO"


@app.route('/contact/getall')
def get_all_contacts():
    if request.method == 'GET':
        contacts = contact_handler.get_all_contacts(session)
        return str(contacts).replace("'", "")

@app.route('/invoice/getopen')
def get_open_invoices():
    if request.method == 'GET':
        opened_invoices = invoice_handler.get_all_open(session)
        return str(opened_invoices).replace("'", "")

@app.route('/invoice/getclose')
def get_closed_invoices():
    if request.method == 'GET':
        closed_invoices = invoice_handler.get_all_closed(session)
        return str(closed_invoices).replace("'", "")


@app.route('/invoice/set_payment/<int:id_invoice>/<string:dt_pagamento>/<float:dolar_pagamento>/<float:valor_pago>')
def testing(id_invoice, dt_pagamento, dolar_pagamento, valor_pago):
    if request.method == 'GET':
        paid = invoice_handler.set_payment(session, id_invoice, dt_pagamento, dolar_pagamento, valor_pago)
        if not paid:
            return "erro"
        else:
            return "ok"

@app.route('/indicator/getdata/high')
def get_high():
    if request.method == 'GET':
        high = f['High']
        high = high[-1]
        return str(high)


@app.route('/strategy/getdata/<int:indicator_id>')
def get_strategy(indicator_id):
    if request.method == 'GET':
        found_strategy = strategy_handler.get_strategy_by_indicator(session, indicator_id)
        if not found_strategy:
            return "erro"
        else:
            req = {}
            req['indicator'] = found_strategy.indicator
            req['days_past'] = found_strategy.days_past
            req['accuracy'] = found_strategy.accuracy
            req['active'] = found_strategy.accuracy
            return str(json.dumps(req)).replace("'", "")

@app.route('/chart/getall/line')
def get_chart():
    if request.method == 'GET':
        line_chart = f['Close']
        return str(line_chart.to_json(orient='table'))

@app.route('/chart/getall/candle')
def get_candle():
    if request.method == 'GET':
        candlestick = f
        return str(f.to_json(orient='table'))

#good buy a fazer para o barraca
@app.route('/overview')
def get_overview():
    if request.method == 'GET':
        return 'False'

@app.route('/indicator/getdata/low')
def get_low():
    if request.method == 'GET':
        low = f['Low']
        low = low[-1]
        return str(low)

@app.route('/invoice/update/<int:nro_invoice>/<string:resp_invoice>/<string:tipo>/<string:dt_emissao>/<string:dt_vencimento>/<string:fornecedor>/<float:valor_invoice>/<float:dolar_provisao>/<string:observacao>')
def update_invoice(nro_invoice, resp_invoice, tipo, dt_emissao, dt_vencimento, fornecedor, valor_invoice, dolar_provisao, observacao):
    if request.method == 'GET':
        try:
            invoice_handler.update_invoice(session, nro_invoice, resp_invoice, tipo, dt_emissao, dt_vencimento, fornecedor, valor_invoice, dolar_provisao, observacao)
            return "OK"
        except:
            return "ERRO"



@app.route('/invoice/getdata/<int:nro_invoice>')
def get_invoice(nro_invoice):
    if request.method == 'GET':
        found_invoice = invoice_handler.get_invoice(session, nro_invoice)
        return str(found_invoice)


@app.route('/invoice/getall')
def invoice_geatll():
    if request.method == 'GET':
        found_invoices = invoice_handler.get_all_invoices(session)
        return str(found_invoices).replace("'", "")

@app.route('/invoice/register/<int:nro_invoice>/<string:resp_invoice>/<string:tipo>/<string:dt_emissao>/<string:dt_vencimento>/<string:fornecedor>/<float:valor_invoice>/<float:dolar_provisao>/<string:observacao>')
def invoice_register(nro_invoice, resp_invoice, tipo, dt_emissao, dt_vencimento, fornecedor, valor_invoice, dolar_provisao, observacao):
    if request.method == 'GET':
        invoice_handler.create_invoice(session, nro_invoice, resp_invoice, tipo, dt_emissao, dt_vencimento, fornecedor, valor_invoice, dolar_provisao, observacao)
        return "ok"


@app.route('/strategy/indicatordays/<int:indicator_id>')
def get_indicator_strategy(indicator_id):
    days = strategy_handler.get_strategy_indicator_days(session, indicator_id)
    return str(days)

@app.route('/invoice/getdata/<int:invoiceid>')
def get_invoice_data():
    invoice = invoice_handler.get_invoice(invoiceid)
    return invoice

@app.route('/indicator/getdata/<int:indicator_id>')
def get_indicator_data(indicator_id):
    indicator = indicator_handler.get_indicator(session, indicator_id)
 #   indicator = indicator.__dict__
    return str(indicator)

@app.route('/indicator/getall')
def get_indicators():
    if request.method == 'GET':
        indicators = indicator_handler.get_all_indicators(session)
        return str(indicators)

@app.route('/notification/getall')
def get_notifications():
    if request.method == 'GET':
        notifications = notification_handler.get_all_notifications(session)
        return str(notifications).replace("'", "")

#@app.route('/invoice/getall')
#def get_invoices():
#    if request.method == 'GET':
#        invoices = invoice_handler.get_all_invoices(session)
#        return str(invoices)

stockchart = StockDataFrame.retype(pd.read_csv('brlusd.csv'))


@app.route('/chart/year/indicator/<int:indicator_id>')
def get_indicator_year_chart(indicator_id):
    yeardata = web.DataReader('BRL=X', 'yahoo')
    stockyear = StockDataFrame.retype(yeardata)
    if request.method == 'GET':
        if indicator_id == 0:
            close = stockyear['close']
            close = close.iloc[-365:]
            return str(close.to_json(orient='table'))
        elif indicator_id == 1:
            macd = stockyear['macd']
            macd = macd.iloc[-365:]
            return str(macd.to_json(orient='table'))
        elif indicator_id == 3:
            macdh = stockyear['macdh']
            macdh = macdh.iloc[-365:]
            return str(macdh.to_json(orient='table'))
        elif indicator_id == 8:
            rsi = stockyear['rsi_6']
            rsi = rsi.iloc[-365:]
            return str(rsi.to_json(orient='table'))
        elif indicator_id == 5:
            bollinger_ub = stockyear['boll_ub']
            bollinger_ub = bollinger_ub.iloc[-365:]
            return str(bollinger_ub.to_json(orient='table'))
        elif indicator_id == 6:
            bollinger_low = stockyear['bollinger_low']
            bollinger_low = bollinger_low.iloc[-365:]
            return str(bollinger_low.to_json(orient='table'))

@app.route('/strategy/getall')
def get_all_strategies():
    found = strategy_handler.get_all(session)
    if not found:
        return "erro"
    else:
        return str(found)

@app.route('/strategy/updateaccuracy/<int:indicator_id>/<int:accuracy>')
def update_strategy_accuracy(indicator_id, accuracy):
    updated = strategy_handler.update_accuracy(session, indicator_id, accuracy)
    if not updated:
        return "erro"
    else:
        return "ok"

@app.route('/strategy/updateflag/<int:indicator_id>/<int:flag>')
def update_strategy_flag(indicator_id, flag):
    updated = strategy_handler.update_flag(session, indicator_id, flag)
    if updated:
        return "ok"
    else:
        return "erro"

@app.route('/chart/month/indicator/<int:indicator_id>')
def get_indicator_month_chart(indicator_id):
    monthdata = web.DataReader('BRL=X', 'yahoo')
    stockmonth = StockDataFrame.retype(monthdata)
    if request.method == 'GET':
        if indicator_id == 0:
            close = stockmonth['close']
            close = close.iloc[-30:]
            return str(close.to_json(orient='table'))
        elif indicator_id == 1:
            macd = stockmonth['macd']
            macd = macd.iloc[-30:]
            return str(macd.to_json(orient='table'))
        elif indicator_id == 3:
            macdh = stockmonth['macdh']
            macdh = macdh.iloc[-30:]
            return str(macdh.to_json(orient='table'))
        elif indicator_id == 8:
            rsi = stockmonth['rsi_6']
            rsi = rsi.iloc[-30:]
            return str(rsi.to_json(orient='table'))
        elif indicator_id == 5:
            bollinger_ub = stockmonth['boll_ub']
            bollinger_ub = bollinger_ub.iloc[-30:]
            return str(bollinger_ub.to_json(orient='table'))
        elif indicator_id == 6:
            bollinger_low = stockmonth['bollinger_low']
            bollinger_low = bollinger_low.iloc[-30:]
            return str(bollinger_low.to_json(orient='table'))

@app.route('/chart/week/indicator/<int:indicator_id>')
def get_indicator_week_chart(indicator_id):
    weekdata = web.DataReader('BRL=X', 'yahoo')
    stockweek = StockDataFrame.retype(weekdata)
    if request.method == 'GET':
        if indicator_id == 1:
            macd = stockweek['macd']
            macd = macd.iloc[-7:]
            return str(macd.to_json(orient='table'))
        elif indicator_id == 0:
            close = stockweek['close']
            close = close.iloc[-7:]
            return str(close.to_json(orient='table'))
        elif indicator_id == 3:
            macdh = stockweek['macdh']
            macdh = macdh.iloc[-7:]
            return str(macdh.to_json(orient='table'))
        elif indicator_id == 8:
            rsi = stockweek['rsi_6']
            rsi = rsi.iloc[-7:]
            return str(rsi.to_json(orient='table'))
        elif indicator_id == 5:
            bollinger_ub = stockweek['boll_ub']
            bollinger_ub = bollinger_ub.iloc[-7:]
            return str(bollinger_ub.to_json(orient='table'))
        elif indicator_id == 6:
            bollinger_low = stockweek['bollinger_low']
            bollinger_low = bollinger_low.iloc[-7:]
            return str(bollinger_low.to_json(orient='table'))


@app.route('/chart/indicator/<int:indicator_id>')
def get_indicator_chart(indicator_id):
    if request.method == 'GET':
        #macd 1, macdh 3, rsi 8, bollinger 4
        if indicator_id == 1:
            macd = stockchart['macd']
            return str(macd.to_json(orient='table'))
        if indicator_id == 3:
            macdh = stockchart['macdh']
            return str(macdh.to_json(orient='table'))
        if indicator_id == 8:
            rsi = stockchart['rsi_6']
            return str(rsi.to_json(orient='table'))
        if indicator_id == 5:
            bollinger_ub = stockchart['boll_ub']
            return str(bollinger_ub.to_json(orient='table'))
        if indicator_id == 6:
            bollinger_low = stockchart['boll_lb']
            return str(bollinger_low.to_json(orient='table'))


@app.route('/signal/getall')
def get_signals():
    if request.method == 'GET':
        signals = signal_handler.get_all_signals(session)
        return str(signals)

app.run(host='0.0.0.0')
