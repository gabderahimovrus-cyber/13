import tempfile
import unittest
from pathlib import Path

from self_mod_bridge import GuardrailViolation, HealthCheckMonitor, HealthSample, SelfModificationBridge


class SelfModificationBridgeTests(unittest.TestCase):
    def test_guardrails_must_be_confirmed_before_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bridge = SelfModificationBridge(root)

            with self.assertRaises(GuardrailViolation):
                bridge.writeFile("notes.txt", "blocked")

            status = bridge.confirm_guardrails()
            self.assertTrue(all(status.values()))
            bridge.writeFile("notes.txt", "allowed")
            self.assertEqual((root / "notes.txt").read_text(encoding="utf-8"), "allowed")

    def test_vfs_rejects_paths_outside_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            bridge = SelfModificationBridge(Path(tmp))
            bridge.confirm_guardrails()

            with self.assertRaises(Exception):
                bridge.vfs.readFile("../outside.txt")

    def test_python_syntax_error_is_rejected_before_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "plugin.py"
            target.write_text("VALUE = 1\n", encoding="utf-8")
            bridge = SelfModificationBridge(root)
            bridge.confirm_guardrails()

            with self.assertRaises(SyntaxError):
                bridge.writeFile("plugin.py", "def broken(:\n    pass\n")

            self.assertEqual(target.read_text(encoding="utf-8"), "VALUE = 1\n")

    def test_snapshot_and_one_word_rollback_restore_latest_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "notes.txt"
            target.write_text("before", encoding="utf-8")
            bridge = SelfModificationBridge(root)
            bridge.confirm_guardrails()

            bridge.writeFile("notes.txt", "after")
            self.assertEqual(target.read_text(encoding="utf-8"), "after")
            bridge.rollback("ROLLBACK")

            self.assertEqual(target.read_text(encoding="utf-8"), "before")

    def test_recursive_loop_guard_rolls_back_session_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bridge = SelfModificationBridge(root)
            bridge.policy.max_iterations_per_request = 1
            bridge.confirm_guardrails()
            session = bridge.new_session()

            bridge.writeFile("a.txt", "first", session)
            with self.assertRaises(GuardrailViolation):
                bridge.writeFile("b.txt", "second", session)

            self.assertEqual((root / "a.txt").read_text(encoding="utf-8"), "")
            self.assertFalse((root / "b.txt").exists())

    def test_execute_only_allows_dependency_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            bridge = SelfModificationBridge(Path(tmp))
            bridge.confirm_guardrails()

            with self.assertRaises(GuardrailViolation):
                bridge.execute(["echo", "nope"])

    def test_context_persistence_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            bridge = SelfModificationBridge(Path(tmp))
            bridge.context.save({"agents": [{"id": "a1", "task": "demo"}]})

            self.assertEqual(bridge.context.load()["agents"][0]["id"], "a1")

    def test_protected_core_file_is_not_writable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.py").write_text("print('core')\n", encoding="utf-8")
            bridge = SelfModificationBridge(root)
            bridge.confirm_guardrails()

            with self.assertRaises(GuardrailViolation):
                bridge.writeFile("main.py", "print('changed')\n")

    def test_health_monitor_rejects_5xx_and_latency_regressions(self):
        monitor = HealthCheckMonitor(baseline_latency_ms=100, max_latency_multiplier=2.0)

        self.assertTrue(monitor.is_healthy([HealthSample(status_code=200, latency_ms=150)]))
        self.assertFalse(monitor.is_healthy([HealthSample(status_code=500, latency_ms=50)]))
        self.assertFalse(monitor.is_healthy([HealthSample(status_code=200, latency_ms=250)]))


if __name__ == "__main__":
    unittest.main()
