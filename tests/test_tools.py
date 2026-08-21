import unittest
from unittest.mock import call, patch

import tools
from tools import open_webpage


class OpenWebpageTests(unittest.TestCase):
    @patch("tools.webbrowser.open", return_value=True)
    def test_opens_http_and_https_urls(self, browser_open):
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

    @patch("tools.webbrowser.open")
    def test_rejects_non_http_hostless_credentialed_or_padded_urls(self, browser_open):
        invalid_urls = (
            "file:///etc/passwd",
            "javascript:alert(1)",
            "https:///missing-host",
            "not-a-url",
            " https://example.com",
            "https://user:secret@example.com",
            "https://example.com\nignored",
        )

        for url in invalid_urls:
            with self.subTest(url=url):
                result = open_webpage.invoke({"url": url})
                self.assertIn("Error", result)

        browser_open.assert_not_called()

    @patch("tools.webbrowser.open", return_value=False)
    def test_reports_when_browser_declines_url(self, browser_open):
        result = open_webpage.invoke({"url": "https://example.com"})

        self.assertIn("Error", result)
        browser_open.assert_called_once_with("https://example.com")

    @patch("tools.webbrowser.open", side_effect=RuntimeError("browser unavailable"))
    def test_reports_browser_errors(self, browser_open):
        result = open_webpage.invoke({"url": "https://example.com"})

        self.assertEqual(result, "Error: browser unavailable")
        browser_open.assert_called_once_with("https://example.com")


class ToolSafetyTests(unittest.TestCase):
    @patch("tools.subprocess.run")
    def test_tts_rejects_invalid_text_before_subprocess(self, run):
        for text in ("", " padded", "line\nfeed", "x" * 301):
            with self.subTest(text=text[:20]):
                result = tools.play_tts_punishment.invoke({"text": text})
                self.assertIn("Error", result)
        run.assert_not_called()

    @patch.object(tools, "ENABLE_WECHAT_ACTIONS", False)
    @patch("tools.subprocess.run")
    def test_wechat_action_fails_closed_without_opt_in(self, run):
        result = tools.send_wechat_shame_message.invoke(
            {"target": "导师", "message": "Please check in with me."}
        )
        self.assertIn("disabled", result)
        run.assert_not_called()

    @patch.object(tools, "ENABLE_APP_TERMINATION", False)
    @patch("tools.subprocess.run")
    def test_app_termination_fails_closed_without_opt_in(self, run):
        result = tools.force_close_app.invoke({"app_name": "Steam"})
        self.assertIn("disabled", result)
        run.assert_not_called()

    def test_chaos_tool_is_a_safe_deprecated_stub(self):
        result = tools.chaos_terminal_punishment.invoke({"message": "anything"})
        self.assertIn("removed", result)

    def test_chaos_tool_is_never_model_visible(self):
        names = {tool.name for tool in tools.ALL_TOOLS}
        self.assertNotIn("chaos_terminal_punishment", names)

    def test_default_registry_contains_only_low_impact_tools(self):
        if not tools.ENABLE_WECHAT_ACTIONS and not tools.ENABLE_APP_TERMINATION:
            self.assertEqual(
                {tool.name for tool in tools.ALL_TOOLS},
                {"play_tts_punishment", "open_webpage", "observe_camera"},
            )


if __name__ == "__main__":
    unittest.main()
