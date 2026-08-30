"""Detect resource.setrlimit()/prlimit() calls that alter process limits."""
import argparse,ast
from pathlib import Path
from maintenance.common import print_failures,production_python_files,require_root
def audit_source(s):
 try:t=ast.parse(s)
 except SyntaxError:return []
 return [f"resource.{n.func.attr}() mutates process limits on line {n.lineno}" for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=="resource" and n.func.attr in {"setrlimit","prlimit"}]
def audit(root):
 root=require_root(root);o=[]
 for r in production_python_files(root):
  try:s=(root/r).read_text(encoding="utf-8")
  except (OSError,UnicodeDecodeError):continue
  o += [f"{r}: {x}" for x in audit_source(s)]
 return o
def main(argv=None):p=argparse.ArgumentParser();p.add_argument("root",nargs="?",default=".");return print_failures(audit(Path(p.parse_args(argv).root)))
if __name__=="__main__":raise SystemExit(main())
