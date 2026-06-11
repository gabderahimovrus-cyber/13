import tempfile
import unittest
from pathlib import Path

from main import Agent, AppState, Message, StateStore, AgentEngine, OllamaClient


class CoreTests(unittest.TestCase):
    def test_state_round_trip_preserves_agent_memory_and_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            store = StateStore(path)
            agent = Agent(
                id="a1",
                name="Тестовый агент",
                role="Проверяет сохранение",
                system_prompt="Говори по-русски",
                goal="Сохранить данные",
            )
            agent.remember_event("важное событие")
            message = Message(sender="Пользователь", recipient="Все", text="Привет")
            agent.remember_message(message)
            state = AppState(agents=[agent], messages=[message], selected_global_model="llama3.1", paused=False)

            store.save(state)
            loaded = store.load()

            self.assertEqual(loaded.selected_global_model, "llama3.1")
            self.assertFalse(loaded.paused)
            self.assertEqual(loaded.agents[0].name, "Тестовый агент")
            self.assertEqual(loaded.agents[0].short_term_memory[0].text, "Привет")
            self.assertIn("важное событие", loaded.agents[0].long_term_memory[0])

    def test_agent_memory_limit_is_enforced(self):
        agent = Agent(id="a1", name="Агент", role="", system_prompt="", goal="", memory_limit=2)
        for index in range(4):
            agent.remember_message(Message(sender="Пользователь", text=f"m{index}"))

        self.assertEqual([message.text for message in agent.short_term_memory], ["m2", "m3"])

    def test_engine_parses_json_embedded_in_markdown(self):
        state = AppState()
        engine = AgentEngine(state, OllamaClient(), threading_lock_for_tests(), lambda _: None, lambda: None)

        parsed = engine._parse_agent_response('```json\n{"respond": false, "message": "", "recipient": "Все"}\n```')

        self.assertFalse(parsed["respond"])
        self.assertEqual(parsed["recipient"], "Все")


def threading_lock_for_tests():
    import threading

    return threading.RLock()


if __name__ == "__main__":
    unittest.main()
