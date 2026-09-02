from maintenance.pandas_pickle_load_audit import audit_source

def test_pandas_pickle_audit_ignores_csv_reads():
    assert audit_source("pd.read_csv(path)\n") == []

def test_pandas_pickle_audit_reports_pickle_reads():
    assert audit_source("pd.read_pickle(path)\n") == ["pd.read_pickle deserializes pickle data on line 1"]
