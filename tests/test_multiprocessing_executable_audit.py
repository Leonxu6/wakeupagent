from maintenance.multiprocessing_executable_audit import audit_source

def test_allows_process_construction(): assert audit_source("multiprocessing.Process(target=work)\n")==[]
def test_reports_child_runtime_mutation(): assert audit_source("multiprocessing.set_executable('/usr/bin/python3')\n")==["multiprocessing.set_executable() changes child runtime policy on line 1"]
