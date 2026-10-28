import os
import unittest
from core.tool_harness import ToolHarness


class TestToolHarness(unittest.TestCase):
    def setUp(self):
        self.session_id = "test_session_123"
        self.harness = ToolHarness(self.session_id)

    def test_write_read_edit_sandbox(self):
        # 1. Write file
        write_res = self.harness.write_file("SPEC.md", "Initial Spec Content")
        self.assertIn("Successfully wrote", write_res)

        # 2. Read file
        content = self.harness.read_file("SPEC.md")
        self.assertEqual(content, "Initial Spec Content")

        # 3. Edit file
        edit_res = self.harness.edit_file("SPEC.md", "Initial", "Updated")
        self.assertIn("Successfully replaced", edit_res)
        updated_content = self.harness.read_file("SPEC.md")
        self.assertEqual(updated_content, "Updated Spec Content")

    def test_path_traversal_prevention(self):
        # Attempt to write outside sandbox
        with self.assertRaises(ValueError):
            self.harness.write_file("../../../secret.txt", "exploit")

    def test_stage_asset(self):
        asset_rel = self.harness.stage_asset("test.png", b"\x89PNG\r\n\x1a\n")
        self.assertTrue(os.path.isfile(os.path.join(self.harness.workspace_dir, asset_rel)))


if __name__ == "__main__":
    unittest.main()
