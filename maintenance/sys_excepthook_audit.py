"""Detect process-wide sys.excepthook replacement."""
import argparse
from pathlib import Path
from maintenance.ast_rules import assignment_targets,iter_assignments
from maintenance.common import print_failures,production_python_files,require_root
def audit_source(s):
 o=[]
 for n in iter_assignments(s):
  if "sys.excepthook" in assignment_targets(n):o.append(f"sys.excepthook replacement mutates process error handling on line {n.lineno}")
 return o
def audit(root):
 root=require_root(root);o=[]
 for r in production_python_files(root):
  try:s=(root/r).read_text(encoding="utf-8")
  except (OSError,UnicodeDecodeError):continue
  o += [f"{r}: {x}" for x in audit_source(s)]
 return o
def main(argv=None):p=argparse.ArgumentParser();p.add_argument("root",nargs="?",default=".");return print_failures(audit(Path(p.parse_args(argv).root)))
if __name__=="__main__":raise SystemExit(main())
