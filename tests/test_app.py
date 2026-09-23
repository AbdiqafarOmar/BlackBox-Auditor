import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app_streamlit.py"


class StreamlitSmokeTests(unittest.TestCase):
    def test_offline_demo_completes_without_ui_exceptions(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=30)
        app.run()
        self.assertEqual(len(app.exception), 0)
        app.button[0].click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.success), 1)
        self.assertIn("112 probes across 2 model(s)", app.success[0].value)
        self.assertGreaterEqual(len(app.dataframe), 4)


if __name__ == "__main__":
    unittest.main()
