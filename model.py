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
    csv_name = Column(String)


class User(Base):

    __tablename__ = 'Users'
    id = Column(Integer, primary_key=True)
    email = Column(String)
    name = Column(String)
    password = Column(String)
    usertype = Column(String)

    def __repr__(self):
        return "<User>(name='%s', usertype='%s')>" %(self.name, self.password,\
                                                     self.usertype)

class Raw_data(Base):

    __tablename__ = "Machine"
    id = Column(Integer, primary_key=True)
    price = Column(Float)
    date = Column(DateTime, server_default=func.now())

class Signal(Base):

    __tablename__ = "Signals"
    id = Column(Integer, primary_key=True)
    indicator = Column(Integer, ForeignKey('Indicator.id'))
    date = Column(DateTime, server_default=func.now())
    accuracy = Column(Float)

class Notification(Base):

    __tablename__ = "Notification"
    id = Column(Integer, primary_key=True)
    platform = Column(String)
    date = Column(DateTime, server_default=func.now())
    message = Column(String)

class Invoice(Base):

    __tablename__ = "Invoice"
    id = Column(Integer, primary_key=True)
    nro_invoice = Column(Integer)
    resp_invoice = Column(String)
    tipo = Column(String)
    dt_emissao = Column(String)
    dt_vencimento = Column(String)
    dt_pagamento = Column(String)
    fornecedor = Column(String)
    valor_invoice = Column(Float)
    dolar_provisao = Column(Float)
    dolar_pagamento = Column(Float)
    valor_pago = Column(Float)
    status = Column(String)
    observacao = Column(String)

class Contact(Base):

    __tablename__ = "Contact"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String)
    phone = Column(String)

class Action(Base):

    __tablename__ = "Action"
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, server_default=func.now())
    performed_by = Column(Integer, ForeignKey(User.id))
    invoice_acted = Column(Integer, ForeignKey(Invoice.id))

class Indicator(Base):
    __tablename__ = "Indicator"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    date = Column(DateTime, server_default=func.now())
    value = Column(Float)

class Strategy(Base):
    __tablename__ = "Strategy"
    id = Column(Integer, primary_key=True)
    indicator = Column(Integer, ForeignKey(Indicator.id))
    days_past = Column(Integer)
    accuracy = Column(Float)



engine = create_engine('postgresql://postgres:postgres@localhost/postgres')
Session = sessionmaker(bind=engine)
Base.metadata.create_all(engine)
