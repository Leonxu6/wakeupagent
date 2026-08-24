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
        check.assert_called_once_with(json_output=False)
        perception.assert_not_called()
        graph.assert_not_called()

    def test_check_json_mode_requests_machine_readable_output(self):
        with patch.object(main, "run_check_mode", return_value=2) as check, \
             patch.object(main, "run_perception_mode") as perception, \
             patch.object(main, "run_graph_mode") as graph:
            status = main.main(["--check-json"])

        self.assertEqual(status, 2)
        check.assert_called_once_with(json_output=True)
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
        for args in (["--graph", "--check"], ["--check", "--check-json"], ["--graph", "--check-json"]):
            with self.subTest(args=args), self.assertRaises(SystemExit):
                parser.parse_args(args)

    def test_message_text_preserves_structured_text_blocks(self):
        self.assertEqual(main._message_text("plain"), "plain")
        self.assertEqual(
            main._message_text([{"type": "text", "text": "first"}, "second", {"type": "image", "url": "x"}]),
            "first second",
        )
        self.assertEqual(
            main._message_text(({"text": "tuple first"}, "tuple second")),
            "tuple first tuple second",
        )
        self.assertEqual(main._message_text({"text": "not-a-list"}), "")

    def test_message_text_normalizes_multiline_and_empty_blocks(self):
        self.assertEqual(main._message_text("  first\nsecond  "), "first second")
        self.assertEqual(
            main._message_text([" first\nline ", {"text": " second\tline "}, {"text": "   "}]),
            "first line second line",
        )

    def test_message_text_bounds_large_strings_and_structured_block_lists(self):
        self.assertEqual(len(main._message_text("x" * (main._MESSAGE_TEXT_LIMIT + 100))), main._MESSAGE_TEXT_LIMIT)
        blocks = [f"part-{i}" for i in range(main._MESSAGE_BLOCK_LIMIT + 5)]
        text = main._message_text(blocks)
        self.assertIn("part-0", text)
        self.assertIn(f"part-{main._MESSAGE_BLOCK_LIMIT - 1}", text)
        self.assertNotIn(f"part-{main._MESSAGE_BLOCK_LIMIT}", text)

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

    def test_ai_message_texts_only_processes_the_bounded_recent_batch(self):
        messages = [SimpleNamespace(type="ai", content=f"ai-{i}") for i in range(main._AI_MESSAGE_LIMIT + 5)]
        texts = main._ai_message_texts({"messages": messages})
        self.assertEqual(len(texts), main._AI_MESSAGE_LIMIT)
        self.assertEqual(texts[0], "ai-5")
        self.assertEqual(texts[-1], f"ai-{main._AI_MESSAGE_LIMIT + 4}")

    def test_runtime_error_text_is_single_line_bounded_and_rich_escaped(self):
        text = main._log_error(RuntimeError("[bold]boom[/bold]\n" + "x" * 1000))
        self.assertIn(r"\[bold]boom\[/bold]", text)
        self.assertNotIn("\n", text)
        self.assertLessEqual(len(text.replace(r"\[", "[").replace(r"\]", "]")), main._ERROR_TEXT_LIMIT)

    def test_runtime_error_rendering_falls_back_when_str_raises(self):
        class Broken:
            def __str__(self):
                raise RuntimeError("render failed")

        self.assertEqual(main._log_error(Broken()), "Broken")

    def test_shutdown_runtime_error_detection_is_narrow(self):
        self.assertTrue(
            main._is_shutdown_runtime_error(RuntimeError("cannot schedule new futures after shutdown"))
        )
        self.assertFalse(main._is_shutdown_runtime_error(RuntimeError("database is locked")))
        self.assertFalse(main._is_shutdown_runtime_error(RuntimeError("shutdown requested")))

    def test_shutdown_runtime_error_detection_tolerates_broken_str(self):
        class BrokenRuntimeError(RuntimeError):
            def __str__(self):
                raise RuntimeError("render failed")

        self.assertFalse(main._is_shutdown_runtime_error(BrokenRuntimeError()))


if __name__ == "__main__":
    unittest.main()
