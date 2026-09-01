from maintenance.numpy_printoptions_audit import audit_source

def test_allows_local_array_formatting(): assert audit_source("text = array2string(values)\n")==[]
def test_reports_process_wide_display_changes(): assert audit_source("np.set_printoptions(precision=3)\n")==["NumPy display configuration changes process-wide output on line 1"]
