from maintenance.unbounded_queue_audit import audit_source

def test_bounded_queue_is_allowed(): assert audit_source("queue.Queue(maxsize=100)\n")==[]
def test_unbounded_queue_is_reported(): assert audit_source("asyncio.Queue()\n")==["unbounded asyncio.Queue() on line 1"]
