import unittest
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


if __name__ == "__main__":
    unittest.main()
