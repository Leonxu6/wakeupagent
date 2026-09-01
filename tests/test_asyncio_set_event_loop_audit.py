from maintenance.asyncio_set_event_loop_audit import audit_source

def test_allows_running_loop_lookup(): assert audit_source("asyncio.get_running_loop()\n")==[]
def test_reports_loop_replacement(): assert audit_source("asyncio.set_event_loop(loop)\n")==["asyncio.set_event_loop() mutates thread-local loop state on line 1"]
