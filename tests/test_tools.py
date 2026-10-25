import os
import subprocess
import unittest
from unittest.mock import call, patch

from tools import open_webpage


class OpenWebpageTests(unittest.TestCase):
    @patch("tools.subprocess.run")
    def test_browser_control_is_disabled_by_default(self, browser_run):
        with patch.dict(os.environ, {}, clear=True):
            result = open_webpage.invoke({"url": "https://example.com"})
        self.assertIn("disabled", result)
        self.assertIn("WAKEUP_ALLOW_BROWSER_CONTROL", result)
        browser_run.assert_not_called()

    @patch("tools.console.print")
    @patch("tools.subprocess.run")
    def test_invalid_browser_flag_is_logged_without_side_effects(self, browser_run, console_print):
        with patch.dict(os.environ, {"WAKEUP_ALLOW_BROWSER_CONTROL": "sometimes"}, clear=True):
            result = open_webpage.invoke({"url": "https://example.com"})
        self.assertIn("disabled", result)
        browser_run.assert_not_called()
        rendered = console_print.call_args.args[0]
        self.assertIn("invalid WAKEUP_ALLOW_BROWSER_CONTROL feature flag", rendered)
        self.assertNotIn("sometimes", rendered)

    @patch("tools.subprocess.run")
    def test_opens_http_and_https_urls(self, browser_run):
        browser_run.return_value = subprocess.CompletedProcess(["osascript"], 0)
        with patch.dict(os.environ, {"WAKEUP_ALLOW_BROWSER_CONTROL": "true"}, clear=True):
            for url in ("http://example.com", "https://example.com/study?q=1"):
                with self.subTest(url=url):
                    result = open_webpage.invoke({"url": url})
                    self.assertIn("已在浏览器中打开", result)

        self.assertEqual(
            browser_run.call_args_list,
            [
                call(
                    ["osascript"],
                    input='open location "http://example.com"\n',
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                ),
                call(
                    ["osascript"],
                    input='open location "https://example.com/study?q=1"\n',
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                ),
            ],
        )

    @patch("tools.console.print")
    @patch("tools.subprocess.run")
    def test_browser_path_markup_is_not_copied_into_logs_or_results(self, browser_run, console_print):
        browser_run.return_value = subprocess.CompletedProcess(["osascript"], 0)
        url = "https://example.com/[red]study[/red]"
        with patch.dict(os.environ, {"WAKEUP_ALLOW_BROWSER_CONTROL": "true"}, clear=True):
            result = open_webpage.invoke({"url": url})
        self.assertIn("https://example.com", result)
        self.assertNotIn("[red]study[/red]", result)
        self.assertEqual(browser_run.call_args.args[0], ["osascript"])
        rendered = console_print.call_args.args[0]
        self.assertNotIn("[red]study[/red]", rendered)

    @patch("tools.console.print")
    @patch("tools.subprocess.run")
    def test_path_query_and_fragment_are_not_copied_into_logs_or_results(self, browser_run, console_print):
        browser_run.return_value = subprocess.CompletedProcess(["osascript"], 0)
        url = "https://example.com/study?token=private#focus"
        with patch.dict(os.environ, {"WAKEUP_ALLOW_BROWSER_CONTROL": "true"}, clear=True):
            result = open_webpage.invoke({"url": url})
        self.assertEqual(browser_run.call_args.args[0], ["osascript"])
        self.assertEqual(browser_run.call_args.kwargs["input"], f'open location "{url}"\n')
        self.assertEqual(result, "已在浏览器中打开：https://example.com")
        self.assertNotIn("study", result)
        self.assertNotIn("token=private", result)
        self.assertNotIn("focus", result)
        rendered = console_print.call_args.args[0]
        self.assertNotIn("study", rendered)
        self.assertNotIn("token=private", rendered)
        self.assertNotIn("#focus", rendered)

    @patch("tools.subprocess.run")
    def test_rejects_non_http_or_hostless_urls_without_side_effects(self, browser_run):
        invalid_urls = (
            "file:///etc/passwd",
            "javascript:alert(1)",
            "https:///missing-host",
            "not-a-url",
        )

        with patch.dict(os.environ, {"WAKEUP_ALLOW_BROWSER_CONTROL": "true"}, clear=True):
            for url in invalid_urls:
                with self.subTest(url=url):
                    result = open_webpage.invoke({"url": url})
                    self.assertIn("Error", result)

        browser_run.assert_not_called()

    @patch("tools.subprocess.run")
    def test_reports_when_browser_declines_url(self, browser_run):
        browser_run.return_value = subprocess.CompletedProcess(["osascript"], 1, stderr="private detail")
        with patch.dict(os.environ, {"WAKEUP_ALLOW_BROWSER_CONTROL": "true"}, clear=True):
            result = open_webpage.invoke({"url": "https://example.com"})

        self.assertIn("Error", result)
        self.assertEqual(browser_run.call_args.args[0], ["osascript"])

    @patch("tools.subprocess.run", side_effect=RuntimeError("private token=https://secret.example"))
    def test_browser_exceptions_do_not_leak_backend_or_path_details(self, browser_run):
        with patch.dict(os.environ, {"WAKEUP_ALLOW_BROWSER_CONTROL": "true"}, clear=True):
            result = open_webpage.invoke({"url": "https://example.com/study?token=private"})

        self.assertEqual(result, "Error: 浏览器打开失败：https://example.com")
        self.assertNotIn("private", result)
        self.assertNotIn("secret.example", result)
        self.assertNotIn("study", result)
        self.assertEqual(
            browser_run.call_args.args[0],
            ["osascript"],
        )

    @patch("tools.subprocess.run", side_effect=FileNotFoundError)
    def test_reports_unsupported_platform_when_open_command_is_missing(self, browser_run):
        with patch.dict(os.environ, {"WAKEUP_ALLOW_BROWSER_CONTROL": "true"}, clear=True):
            result = open_webpage.invoke({"url": "https://example.com"})

        self.assertEqual(result, "Error: osascript 命令不存在（仅支持 macOS）")
        browser_run.assert_called_once()

    @patch("tools.subprocess.run")
    def test_browser_launch_timeout_is_bounded_and_redacted(self, browser_run):
        browser_run.side_effect = subprocess.TimeoutExpired(["open"], 10)
        with patch.dict(os.environ, {"WAKEUP_ALLOW_BROWSER_CONTROL": "true"}, clear=True):
            result = open_webpage.invoke({"url": "https://example.com/private?token=secret"})

        self.assertEqual(result, "Error: 浏览器打开超时：https://example.com")
        browser_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
