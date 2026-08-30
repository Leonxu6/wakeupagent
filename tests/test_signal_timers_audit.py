from maintenance.signal_timers_audit import audit_source

def test_signal_reads_are_allowed(): assert audit_source("signal.getsignal(signal.SIGTERM)\n")==[]
def test_alarm_is_reported(): assert audit_source("signal.alarm(5)\n")==["signal.alarm() mutates process timers on line 1"]
