from maintenance.decimal_default_context_audit import audit_source

def test_allows_local_contexts(): assert audit_source("ctx=decimal.Context(prec=28)\n")==[]
def test_reports_global_default_mutation(): assert audit_source("decimal.DefaultContext.prec=12\n")==["decimal.DefaultContext mutation changes process Decimal defaults on line 1"]
