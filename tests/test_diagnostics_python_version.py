import unittest

import diagnostics


class PythonVersionProbeTests(unittest.TestCase):
    def test_accepts_nonnegative_major_minor_tuples(self):
        self.assertEqual(diagnostics._version_pair((3, 12)), (3, 12))
        self.assertEqual(diagnostics._version_pair((4, 0)), (4, 0))

    def test_rejects_malformed_version_probes(self):
        invalid = ((3,), (3, 12, 1), [3, 12], (True, 12), (3, -1), ("3", 12))
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                diagnostics._version_pair(value)


if __name__ == "__main__":
    unittest.main()
