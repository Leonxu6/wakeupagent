from pathlib import Path

from maintenance.python_compile_audit import audit_python_sources, python_files


def test_python_compile_audit_detects_syntax_errors(tmp_path: Path):
    (tmp_path / "good.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "bad.py").write_text("if True print('x')\n", encoding="utf-8")
    failures = audit_python_sources(tmp_path)
    assert len(failures) == 1
    assert failures[0].startswith("bad.py:")


def test_python_files_skip_generated_directories(tmp_path: Path):
    (tmp_path / "src.py").write_text("pass\n", encoding="utf-8")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "ignored.py").write_text("bad syntax !!!\n", encoding="utf-8")
    assert python_files(tmp_path) == [tmp_path / "src.py"]
