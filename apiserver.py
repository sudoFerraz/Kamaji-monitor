#Flask API server

import auxiliary
from flask import Flask, render_template
from flask import jsonify
from flask import request
import json
import pandas as pd
from stockstats import StockDataFrame
import model
from model import User, Notification, Action, Signal, Raw_data, Indicator
import flask
from flask_sqlalchemy import SQLAlchemy
import sqlalchemy
from flask_admin import Admin
import pandas_datareader as web

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

@app.route('/contact/delete/<string:email>')
def delete_contact(email):
    if request.method == 'GET':
        try:
            contact_handler.delete_contact(session, email)
            return "OK"
        except:
            return "ERRO"

@app.route('/contact/register/<string:name>/<string:email>/<string:phone>')
def register_contact(name, email, phone):
    if request.method == 'GET':
        try:
            new_contact = contact_handler.create_contact(session, name, email, phone)
            return "OK"
        except:
            return "ERRO"

@app.route('/strategy/getall')
def get_all_strategy():
    if request.method == 'GET':
        return "ok"

@app.route('/contact/getall')
def get_all_contacts():
    if request.method == 'GET':
        contacts = contact_handler.get_all_contacts(session)
        return str(contacts)

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


@app.route('/invoice/set_payment/<int:nro_invoice>/<string:dt_pagamento>/<float:dolar_pagamento>/<float:valor_pago>')
def testing(nro_invoice, dt_pagamento, dolar_pagamento, valor_pago):
    if request.method == 'GET':
        try:
            invoice_handler.set_payment(session, nro_invoice, dt_pagamento, dolar_pagamento, valor_pago)
            return 'OK'
        except:
            return 'ERRO'

@app.route('/indicator/getdata/high')
def get_high():
    if request.method == 'GET':
        high = f['High']
        high = high[-1]
        return str(high)


@app.route('/strategy/<int:indicator_id>')
def get_strategy():
    if request.method == 'GET':
        pass

@app.route('/chart/getall/line')
def get_chart():
    if request.method == 'GET':
        line_chart = f['Close']
        return str(line_chart.to_json(orient='table'))

#good buy a fazer para o barraca
@app.route('/overview')
def get_overview():
    if request.method == 'GET':
        return 'True'

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

@app.route('/strategy/indicator_days/<int:indicator_id>')
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
        return str(notifications)

#@app.route('/invoice/getall')
#def get_invoices():
#    if request.method == 'GET':
#        invoices = invoice_handler.get_all_invoices(session)
#        return str(invoices)

stockchart = StockDataFrame.retype(pd.read_csv('brlusd.csv'))

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
            bollinger = stockchart['boll_ub']
            return str(bollinger.to_json(orient='table'))
        if indicator_id == 6:
            bollinger_low = stockchart['boll_lb']


@app.route('/signal/getall')
def get_signals():
    if request.method == 'GET':
        signals = signal_handler.get_all_signals(session)
        return str(signals)

app.run(host='0.0.0.0')
