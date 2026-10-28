import unittest
from core.stateful_runner import parse_tool_calls


class TestToolParser(unittest.TestCase):
    def test_parse_standard_tool_call(self):
        text = """Here is my plan to create the animation:
<tool_call>
<function=Write>
<parameter=path>index.html</parameter>
<parameter=content><!DOCTYPE html><html><body><canvas id="c"></canvas></body></html></parameter>
</tool_call>
Now let's view it."""
        tools = parse_tool_calls(text)
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0][0], "Write")
        self.assertEqual(tools[0][1]["path"], "index.html")
        self.assertIn("canvas id", tools[0][1]["content"])

    def test_parse_multiple_tool_calls(self):
        text = """
<tool_call>
<function=Write>
<parameter=path>SPEC.md</parameter>
<parameter=content>Format: 1280x720, 30fps</parameter>
</tool_call>
<tool_call>
<function=Write>
<parameter=path>index.html</parameter>
<parameter=content><html><body>Hello</body></html></parameter>
</tool_call>
"""
        tools = parse_tool_calls(text)
        self.assertEqual(len(tools), 2)
        self.assertEqual(tools[0][0], "Write")
        self.assertEqual(tools[0][1]["path"], "SPEC.md")
        self.assertEqual(tools[1][0], "Write")
        self.assertEqual(tools[1][1]["path"], "index.html")

    def test_parse_markdown_code_fallback(self):
        text = """Here is the code:
```html
<!DOCTYPE html>
<html>
  <head><script>window.READY=true; window.seek=t=>{};</script></head>
  <body></body>
</html>
```"""
        tools = parse_tool_calls(text)
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0][0], "Write")
        self.assertEqual(tools[0][1]["path"], "index.html")
        self.assertIn("window.READY=true", tools[0][1]["content"])


if __name__ == "__main__":
    unittest.main()
