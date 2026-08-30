"""Detect asyncio.create_task() without a task name for observability."""
import argparse,ast
from pathlib import Path
from maintenance.common import print_failures,production_python_files,require_root
def audit_source(s):
 try:t=ast.parse(s)
 except SyntaxError:return []
 return [f"asyncio.create_task() without name on line {n.lineno}" for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=="asyncio" and n.func.attr=="create_task" and not any(k.arg=="name" for k in n.keywords)]
def audit(root):
 root=require_root(root);o=[]
 for r in production_python_files(root):
  try:s=(root/r).read_text(encoding="utf-8")
  except (OSError,UnicodeDecodeError):continue
  o += [f"{r}: {x}" for x in audit_source(s)]
 return o
def main(argv=None):p=argparse.ArgumentParser();p.add_argument("root",nargs="?",default=".");return print_failures(audit(Path(p.parse_args(argv).root)))
if __name__=="__main__":raise SystemExit(main())
