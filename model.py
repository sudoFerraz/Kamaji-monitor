#Database Modeling
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Boolean, Date, DateTime, Float
from sqlalchemy import ForeignKey, LargeBinary
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func

Base = declarative_base()

class CSV(Base):
    __tablename__ = 'CSVs'
    id = Column(Integer, primary_key=True)
    csv_file = Column(LargeBinary)
    csv_name = Column(String(99))
"""
class Relatorios(Base):
    __tablename__ = 'Relatorios'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    csv_id = Column(Integer, ForeignKey('CSV.id'))
    download = Column(String)
"""

class Strategy_type(Base):
    __tablename__ = 'Strategy_type'
    id = Column(Integer, primary_key=True)
    trade_type = Column(Integer)

class Users(Base):

    __tablename__ = 'Users'
    id = Column(Integer, primary_key=True)
    email = Column(String(99))
    name = Column(String(99))
    password = Column(String(99))
    usertype = Column(String(99))


class Raw_data(Base):

    __tablename__ = "Machine"
    id = Column(Integer, primary_key=True)
    price = Column(Float)
    date = Column(DateTime, server_default=func.now())

class Tendence(Base):

    __tablename__ = "Tendence"
    id = Column(Integer, primary_key=True)
    flag = Column(Boolean)
    date = Column(DateTime, server_default=func.now())

class Sugestion(Base):

    __tablename__ = "Sugestion"
    id = Column(Integer, primary_key=True)
    sugestion = Column(String(99))
    invoice_id = Column(Integer)
    sugestion_date = Column(DateTime, server_default=func.now())

class Signal(Base):

    __tablename__ = "Signals"
    id = Column(Integer, primary_key=True)
    indicator = Column(Integer, ForeignKey('Indicator.id'))
    date = Column(DateTime, server_default=func.now())
    accuracy = Column(Float)

class Notification(Base):

    __tablename__ = "Notification"
    id = Column(Integer, primary_key=True)
    platform = Column(String(99))
    date = Column(DateTime, server_default=func.now())
    message = Column(String(99))

class Forecast(Base):

    __tablename__ = "Forecast"
    id = Column(Integer, primary_key=True)
    intervalo = Column(Integer)
    accuracy = Column(Float)
    previsao = Column(Integer)
    modelo = Column(String(99))
    invoice_id = Column(String(99))

class Invoice(Base):

    __tablename__ = "Invoice"
    id = Column(Integer, primary_key=True)
    nro_invoice = Column(String(99))
    resp_invoice = Column(String(99))
    tipo = Column(String(99))
    dt_emissao = Column(String(99))
    dt_vencimento = Column(String(99))
    dt_pagamento = Column(String(99))
    fornecedor = Column(String(99))
    valor_invoice = Column(Float)
    dolar_provisao = Column(Float)
    dolar_pagamento = Column(Float)
    valor_pago = Column(Float)
    status = Column(String(99))
    observacao = Column(String(300))
    imposto = Column(Float)

class Contact(Base):

    __tablename__ = "Contact"
    id = Column(Integer, primary_key=True)
    name = Column(String(99))
    email = Column(String(99))
    phone = Column(String(99))


class Indicator(Base):
    __tablename__ = "Indicator"
    id = Column(Integer, primary_key=True)
    name = Column(String(99))
    date = Column(DateTime, server_default=func.now())
    value = Column(Float)

#Resetar tabela
class Strategy(Base):
    __tablename__ = "Strategy"
    id = Column(Integer, primary_key=True)
    indicator = Column(Integer, ForeignKey(Indicator.id))
    days_past = Column(Integer)
    accuracy = Column(Float)
    active = Column(Boolean)


class Action(Base):

    __tablename__ = "Action"
    id = Column(Integer, primary_key=True)
    date_time = Column(DateTime, server_default=func.now())
    performed_by = Column(Integer, ForeignKey(Users.id))
    invoice_acted = Column(Integer, ForeignKey(Invoice.id))
    action_type = Column(String(99))
    notification_acted = Column(Integer, ForeignKey(Notification.id))
    strategy_acted = Column(Integer, ForeignKey(Strategy.id))



engine = create_engine('mysql+pymysql://kamaji_user:kamaji2018@localhost/kamaji')
Session = sessionmaker(bind=engine)
Base.metadata.create_all(engine)
