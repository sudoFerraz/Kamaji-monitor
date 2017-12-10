import auxiliary
import model
from send_email import SendMessage as send

notification_handler = auxiliary.notification_handler()
signal_handler = auxiliary.signal_handler()
session = auxiliary.ostools().db_connection()
strategy_handler = auxiliary.strategy_handler()


def check_triggers():
	rsi_trigger = False
	macd_trigger = False
	macdh_trigger = False
	bollinger_lb_trigger = False
	bollinger_ub_trigger = False

	rsi_signal = signal_handler.get_signal_by_indicator(session, 8)
	macd_signal = signal_handler.get_signal_by_indicator(session, 1)
	macdh_signal = signal_handler.get_signal_by_indicator(session, 3)
	bollinger_ub_signal = signal_handler.get_signal_by_indicator(session, 5)
	bollinger_lb_signal = signal_handler.get_signal_by_indicator(session, 6)

	rsi_strategy = strategy_handler.get_strategy_by_indicator(session, 8)
	macd_strategy = signal_handler.get_strategy_by_indicator(session, 1)
	macdh_strategy = signal_handler.get_signal_by_indicator(session, 3)
	bollinger_lb_strategy = signal_handler.get_signal_by_indicator(session, 5)
	bollinger_ub_strategy = signal_handler.get_signal_by_indicator(session, 6)

	if rsi_signal:
		if rsi_signal.accuracy >= rsi_strategy.accuracy:
			rsi_trigger = True
		else:
			rsi_trigger = False
	else:
		rsi_trigger = False

	if macd_signal:
		if macd_signal.accuracy >= macd_strategy.accuracy:
			macd_trigger = True
		else:
			macd_trigger = False
	else:
		macd_trigger = False

	if macdh_signal:
		if macdh_signal.accuracy >= macdh_strategy.accuracy:
			macdh_trigger = True
		else:
			macdh_trigger = False
	else:
		macdh_trigger = False

	if bollinger_ub_signal:
		if bollinger_ub_signal.accuracy >= bollinger_ub_strategy.accuracy:
			bollinger_ub_trigger = True
		else:
			bollinger_ub_trigger = False
	else:
		bollinger_ub_trigger = False

	if bollinger_lb_signal:
		if bollinger_lb_signal.accuracy >= bollinger_lb_strategy.accuracy:
			bollinger_lb_trigger = True
		else:
			bollinger_lb_trigger = False
	else:
		bollinger_lb_trigger = False
		



