import unittest

import diagnostics


class DiagnosticsRenderingTests(unittest.TestCase):
    def test_single_line_falls_back_to_type_name_when_str_raises(self):
        class Broken:
            def __str__(self):
                raise RuntimeError("render failed")

        self.assertEqual(diagnostics._single_line(Broken()), "Broken")

    def test_format_checks_keeps_broken_details_machine_readable(self):
        class Broken:
            def __str__(self):
                raise RuntimeError("render failed")

        text = diagnostics.format_checks([diagnostics.Check("runtime", False, Broken())])  # type: ignore[arg-type]
        self.assertEqual(text, "[WARN] runtime: Broken")


if __name__ == "__main__":
    unittest.main()
