from maintenance.unbounded_deque_audit import audit_source

def test_bounded_deque_is_allowed(): assert audit_source("collections.deque(maxlen=128)\n")==[]
def test_unbounded_deque_is_reported(): assert audit_source("collections.deque()\n")==["collections.deque() without maxlen on line 1"]
