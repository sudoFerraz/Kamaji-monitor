import auxiliary
import model
from datetime import datetime
from datetime import timedelta
from datetime import date
import pandas as pd

def forecast():

	session = auxiliary.ostools().db_connection()

	abertas = session.query(model.Invoice).filter_by(status="aberta").all()
	count = len(abertas)
	intervalos = []
	modelos = []
	ids = []
	for invoice in abertas:
	    ids.append(invoice.id)
	    modelos.append("svm")
	    vencimento = datetime.strptime(invoice.dt_vencimento, '%Y-%m-%d')
	    now = datetime.now()
	    intervalo = vencimento - now
	    intervalo = abs(intervalo.days)
	    intervalos.append(intervalo)

	df = pd.DataFrame({"ID":ids, "Interval":intervalos, "Model":modelos})
	df.to_csv("invoices_forecast.csv", mode="w", header=True)
