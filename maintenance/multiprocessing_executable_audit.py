"""Detect process-wide multiprocessing executable mutation."""
import argparse
from pathlib import Path
from maintenance.ast_rules import call_name,iter_calls
from maintenance.common import print_failures,production_python_files,require_root
def audit_source(s): return [f"multiprocessing.set_executable() changes child runtime policy on line {c.lineno}" for c in iter_calls(s) if call_name(c)=="multiprocessing.set_executable"]
def audit(root):
 root=require_root(root);o=[]
 for r in production_python_files(root):
  try:s=(root/r).read_text(encoding="utf-8")
  except (OSError,UnicodeDecodeError):continue
  o += [f"{r}: {x}" for x in audit_source(s)]
 return o
def main(argv=None):p=argparse.ArgumentParser();p.add_argument("root",nargs="?",default=".");return print_failures(audit(Path(p.parse_args(argv).root)))
if __name__=="__main__":raise SystemExit(main())
