"""Detect sys.settrace() runtime tracing hooks."""
import argparse
from pathlib import Path
from maintenance.ast_rules import call_name,iter_calls
from maintenance.common import print_failures,production_python_files,require_root
def audit_source(s): return [f"sys.settrace() installs a runtime tracing hook on line {c.lineno}" for c in iter_calls(s) if call_name(c)=="sys.settrace"]
def audit(root):
 root=require_root(root);o=[]
 for r in production_python_files(root):
  try:s=(root/r).read_text(encoding="utf-8")
  except (OSError,UnicodeDecodeError):continue
  o += [f"{r}: {x}" for x in audit_source(s)]
 return o
def main(argv=None):p=argparse.ArgumentParser();p.add_argument("root",nargs="?",default=".");return print_failures(audit(Path(p.parse_args(argv).root)))
if __name__=="__main__":raise SystemExit(main())
