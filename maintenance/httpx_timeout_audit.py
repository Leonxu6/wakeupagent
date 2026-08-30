"""Detect common httpx calls without an explicit timeout policy."""
import argparse,ast
from pathlib import Path
from maintenance.common import print_failures,production_python_files,require_root
T={"get","post","put","patch","delete","request","Client","AsyncClient"}
def audit_source(s):
 try:t=ast.parse(s)
 except SyntaxError:return []
 o=[]
 for n in ast.walk(t):
  f=n.func if isinstance(n,ast.Call) else None
  if isinstance(f,ast.Attribute) and isinstance(f.value,ast.Name) and f.value.id=="httpx" and f.attr in T and not any(k.arg=="timeout" for k in n.keywords):o.append(f"httpx.{f.attr}() without explicit timeout on line {n.lineno}")
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
