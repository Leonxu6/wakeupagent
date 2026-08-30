from maintenance.numpy_global_state_audit import audit_source

def test_local_numpy_work_is_allowed(): assert audit_source("np.asarray(values)\n")==[]
def test_numpy_state_mutation_is_reported(): assert audit_source("np.seterr(all='raise')\n")==["NumPy global state mutation via seterr() on line 1"]
