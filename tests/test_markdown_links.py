from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "maintenance" / "markdown_links.py"
spec = spec_from_file_location("markdown_links", MODULE_PATH)
module = module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


def test_checker_accepts_existing_local_and_external_links(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "other.md").write_text("# Other\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("[local](docs/other.md) [web](https://example.com)\n", encoding="utf-8")
    assert module.broken_local_links(tmp_path) == []


def test_checker_reports_missing_local_target_and_keeps_anchor_text(tmp_path):
    (tmp_path / "README.md").write_text("[missing](docs/nope.md#section)\n", encoding="utf-8")
    broken = module.broken_local_links(tmp_path)
    assert [(item.source.as_posix(), item.target) for item in broken] == [("README.md", "docs/nope.md#section")]
