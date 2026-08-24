import os
import unittest
from unittest.mock import call, patch

from tools import open_webpage


class OpenWebpageTests(unittest.TestCase):
    @patch("tools.webbrowser.open")
    def test_browser_control_is_disabled_by_default(self, browser_open):
        with patch.dict(os.environ, {}, clear=True):
            result = open_webpage.invoke({"url": "https://example.com"})
        self.assertIn("disabled", result)
        self.assertIn("WAKEUP_ALLOW_BROWSER_CONTROL", result)
        browser_open.assert_not_called()

    @patch("tools.webbrowser.open", return_value=True)
    def test_opens_http_and_https_urls(self, browser_open):
        with patch.dict(os.environ, {"WAKEUP_ALLOW_BROWSER_CONTROL": "true"}, clear=True):
            for url in ("http://example.com", "https://example.com/study?q=1"):
                with self.subTest(url=url):
                    result = open_webpage.invoke({"url": url})
                    self.assertIn("已在浏览器中打开", result)

        self.assertEqual(
            browser_open.call_args_list,
            [
                call("http://example.com"),
                call("https://example.com/study?q=1"),
            ],
        )

    @patch("tools.console.print")
    @patch("tools.webbrowser.open", return_value=True)
    def test_escapes_rich_markup_in_browser_logs(self, browser_open, console_print):
        url = "https://example.com/[red]study[/red]"
        with patch.dict(os.environ, {"WAKEUP_ALLOW_BROWSER_CONTROL": "true"}, clear=True):
            result = open_webpage.invoke({"url": url})
        self.assertIn(url, result)
        browser_open.assert_called_once_with(url)
        rendered = console_print.call_args.args[0]
        self.assertIn(r"\[red]study\[/red]", rendered)

    @patch("tools.console.print")
    @patch("tools.webbrowser.open", return_value=True)
    def test_query_and_fragment_are_not_copied_into_logs_or_results(self, browser_open, console_print):
        url = "https://example.com/study?token=private#focus"
        with patch.dict(os.environ, {"WAKEUP_ALLOW_BROWSER_CONTROL": "true"}, clear=True):
            result = open_webpage.invoke({"url": url})
        browser_open.assert_called_once_with(url)
        self.assertIn("https://example.com/study", result)
        self.assertNotIn("token=private", result)
        self.assertNotIn("focus", result)
        rendered = console_print.call_args.args[0]
        self.assertNotIn("token=private", rendered)
        self.assertNotIn("#focus", rendered)

    @patch("tools.webbrowser.open")
    def test_rejects_non_http_or_hostless_urls_without_side_effects(self, browser_open):
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

        browser_open.assert_not_called()

    @patch("tools.webbrowser.open", return_value=False)
    def test_reports_when_browser_declines_url(self, browser_open):
        with patch.dict(os.environ, {"WAKEUP_ALLOW_BROWSER_CONTROL": "true"}, clear=True):
            result = open_webpage.invoke({"url": "https://example.com"})

        self.assertIn("Error", result)
        browser_open.assert_called_once_with("https://example.com")

    @patch("tools.webbrowser.open", side_effect=RuntimeError("browser unavailable"))
    def test_reports_browser_errors(self, browser_open):
        with patch.dict(os.environ, {"WAKEUP_ALLOW_BROWSER_CONTROL": "true"}, clear=True):
            result = open_webpage.invoke({"url": "https://example.com"})

        self.assertEqual(result, "Error: browser unavailable")
        browser_open.assert_called_once_with("https://example.com")


if __name__ == "__main__":
    unittest.main()
