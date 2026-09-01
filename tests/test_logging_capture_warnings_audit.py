from maintenance.logging_capture_warnings_audit import audit_source

def test_allows_regular_logging(): assert audit_source("logging.getLogger(__name__)\n")==[]
def test_reports_global_routing_changes(): assert audit_source("logging.captureWarnings(True)\n")==["logging.captureWarnings() mutates process warning routing on line 1"]
