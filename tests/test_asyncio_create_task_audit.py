from maintenance.asyncio_create_task_audit import audit_source

def test_named_task_is_allowed(): assert audit_source("asyncio.create_task(worker(), name='worker')\n")==[]
def test_anonymous_task_is_reported(): assert audit_source("asyncio.create_task(worker())\n")==["asyncio.create_task() without name on line 1"]
