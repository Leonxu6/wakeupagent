"""Detect NumPy calls that alter process-wide numerical/display policy."""
import argparse,ast
from pathlib import Path
from maintenance.common import print_failures,production_python_files,require_root
M={"seterr","seterrcall","set_printoptions"}
def audit_source(s):
 try:t=ast.parse(s)
 except SyntaxError:return []
 return [f"NumPy global state mutation via {n.func.attr}() on line {n.lineno}" for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id in {"np","numpy"} and n.func.attr in M]
def audit(root):
 root=require_root(root);o=[]
 for r in production_python_files(root):
  try:s=(root/r).read_text(encoding="utf-8")
  except (OSError,UnicodeDecodeError):continue
  o += [f"{r}: {x}" for x in audit_source(s)]
 return o
def main(argv=None):p=argparse.ArgumentParser();p.add_argument("root",nargs="?",default=".");return print_failures(audit(Path(p.parse_args(argv).root)))
if __name__=="__main__":raise SystemExit(main())
