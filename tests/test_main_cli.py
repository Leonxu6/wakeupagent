import unittest
from types import SimpleNamespace
from unittest.mock import patch

import main


class MainCliTests(unittest.TestCase):
    def test_check_mode_returns_diagnostics_status_without_starting_runtime(self):
        with patch.object(main, "run_check_mode", return_value=3) as check, \
             patch.object(main, "run_perception_mode") as perception, \
             patch.object(main, "run_graph_mode") as graph:
            status = main.main(["--check"])

        self.assertEqual(status, 3)
        check.assert_called_once_with()
        perception.assert_not_called()
        graph.assert_not_called()

    def test_graph_mode_runs_only_graph(self):
        with patch.object(main, "run_graph_mode") as graph, patch.object(main, "run_perception_mode") as perception:
            status = main.main(["--graph"])
        self.assertEqual(status, 0)
        graph.assert_called_once_with()
        perception.assert_not_called()

    def test_default_mode_starts_perception(self):
        with patch.object(main, "run_graph_mode") as graph, patch.object(main, "run_perception_mode") as perception:
            status = main.main([])
        self.assertEqual(status, 0)
        perception.assert_called_once_with()
        graph.assert_not_called()

    def test_modes_are_mutually_exclusive(self):
        parser = main.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--graph", "--check"])

    def test_message_text_preserves_structured_text_blocks(self):
        self.assertEqual(main._message_text("plain"), "plain")
        self.assertEqual(
            main._message_text([{"type": "text", "text": "first"}, "second", {"type": "image", "url": "x"}]),
            "first second",
        )
        self.assertEqual(main._message_text({"text": "not-a-list"}), "")

    def test_message_text_normalizes_multiline_and_empty_blocks(self):
        self.assertEqual(main._message_text("  first\nsecond  "), "first second")
        self.assertEqual(
            main._message_text([" first\nline ", {"text": " second\tline "}, {"text": "   "}]),
            "first line second line",
        )

    def test_ai_message_texts_tolerates_missing_and_malformed_batches(self):
        for output in (None, [], {}, {"messages": None}, {"messages": "not-a-list"}):
            with self.subTest(output=output):
                self.assertEqual(main._ai_message_texts(output), [])

        output = {
            "messages": [
                SimpleNamespace(type="human", content="ignore"),
                SimpleNamespace(type="ai", content=" first\nanswer "),
                SimpleNamespace(type="ai", content=[{"text": "second"}]),
                SimpleNamespace(type="ai", content=None),
            ]
        }
        self.assertEqual(main._ai_message_texts(output), ["first answer", "second"])

    def test_shutdown_runtime_error_detection_is_narrow(self):
        self.assertTrue(
            main._is_shutdown_runtime_error(RuntimeError("cannot schedule new futures after shutdown"))
        )
        self.assertFalse(main._is_shutdown_runtime_error(RuntimeError("database is locked")))
        self.assertFalse(main._is_shutdown_runtime_error(RuntimeError("shutdown requested")))


if __name__ == "__main__":
    unittest.main()
