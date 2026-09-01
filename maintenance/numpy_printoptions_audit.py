"""Detect NumPy display configuration changes that affect the whole process."""
import argparse
from pathlib import Path
from maintenance.ast_rules import call_name, iter_calls
from maintenance.common import print_failures, production_python_files, require_root

def audit_source(source):
    return [f"NumPy display configuration changes process-wide output on line {call.lineno}" for call in iter_calls(source) if call_name(call) in {"numpy.set_printoptions", "np.set_printoptions"}]

def audit(root):
    root = require_root(root); findings = []
    for rel in production_python_files(root):
        try: source = (root / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError): continue
        findings += [f"{rel}: {item}" for item in audit_source(source)]
    return findings

def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument("root", nargs="?", default="."); return print_failures(audit(Path(parser.parse_args(argv).root)))
if __name__ == "__main__": raise SystemExit(main())
