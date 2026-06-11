"""Русскоязычная локальная многоагентная чат-система для Windows.

Приложение использует Tkinter для настольного интерфейса и Ollama для локальных
языковых моделей. Все данные сохраняются локально в JSON и восстанавливаются при
следующем запуске.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_DOWNLOAD_URL = "https://ollama.com/download/windows"
ENGINE_INTERVAL_SECONDS = 5
MAX_CHAT_CONTEXT = 30
MAX_SHORT_TERM_MEMORY = 20


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def app_data_dir() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    path = Path(base) / "LocalMultiAgentChat"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class Message:
    sender: str
    text: str
    recipient: str = "Все"
    created_at: str = field(default_factory=now_iso)
    kind: str = "message"


@dataclass
class Agent:
    id: str
    name: str
    role: str
    system_prompt: str
    goal: str
    model: str = ""
    active: bool = True
    memory_limit: int = 20
    short_term_memory: list[Message] = field(default_factory=list)
    long_term_memory: list[str] = field(default_factory=list)

    def remember_message(self, message: Message) -> None:
        self.short_term_memory.append(message)
        limit = max(1, self.memory_limit or MAX_SHORT_TERM_MEMORY)
        self.short_term_memory = self.short_term_memory[-limit:]

    def remember_event(self, event: str) -> None:
        if event.strip():
            self.long_term_memory.append(f"{now_iso()} — {event.strip()}")
            self.long_term_memory = self.long_term_memory[-200:]


@dataclass
class AppState:
    agents: list[Agent] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    selected_global_model: str = ""
    paused: bool = True


class StateStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (app_data_dir() / "state.json")

    def load(self) -> AppState:
        if not self.path.exists():
            return self._default_state()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return AppState(
                agents=[self._agent_from_dict(item) for item in raw.get("agents", [])],
                messages=[Message(**item) for item in raw.get("messages", [])],
                events=list(raw.get("events", [])),
                selected_global_model=raw.get("selected_global_model", ""),
                paused=bool(raw.get("paused", True)),
            )
        except Exception as exc:  # noqa: BLE001 - startup recovery must not crash UI
            backup = self.path.with_suffix(f".broken-{int(time.time())}.json")
            self.path.replace(backup)
            state = self._default_state()
            state.events.append(f"{now_iso()} — Не удалось загрузить состояние: {exc}. Создана резервная копия {backup}.")
            return state

    def save(self, state: AppState) -> None:
        serializable = {
            "agents": [self._agent_to_dict(agent) for agent in state.agents],
            "messages": [asdict(message) for message in state.messages[-1000:]],
            "events": state.events[-2000:],
            "selected_global_model": state.selected_global_model,
            "paused": state.paused,
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def _default_state(self) -> AppState:
        return AppState(
            agents=[
                Agent(
                    id=str(uuid.uuid4()),
                    name="Аналитик",
                    role="Анализирует сообщения и формулирует выводы",
                    system_prompt="Ты автономный ИИ-агент. Отвечай кратко, по делу и на русском языке.",
                    goal="Помогать пользователю и другим агентам анализировать ситуацию.",
                ),
                Agent(
                    id=str(uuid.uuid4()),
                    name="Координатор",
                    role="Координирует взаимодействие агентов",
                    system_prompt="Ты следишь за ходом обсуждения, предлагаешь следующие шаги и можешь обращаться к другим агентам.",
                    goal="Поддерживать продуктивный многоагентный диалог.",
                ),
            ],
            messages=[Message(sender="Система", text="Добро пожаловать! Установите/запустите Ollama, выберите модель и снимите систему с паузы.", kind="system")],
            events=[f"{now_iso()} — Создано начальное состояние."],
            paused=True,
        )

    def _agent_from_dict(self, data: dict[str, Any]) -> Agent:
        return Agent(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", "Новый агент"),
            role=data.get("role", ""),
            system_prompt=data.get("system_prompt", ""),
            goal=data.get("goal", ""),
            model=data.get("model", ""),
            active=bool(data.get("active", True)),
            memory_limit=int(data.get("memory_limit", MAX_SHORT_TERM_MEMORY) or MAX_SHORT_TERM_MEMORY),
            short_term_memory=[Message(**item) for item in data.get("short_term_memory", [])],
            long_term_memory=list(data.get("long_term_memory", [])),
        )

    def _agent_to_dict(self, agent: Agent) -> dict[str, Any]:
        data = asdict(agent)
        data["short_term_memory"] = [asdict(message) for message in agent.short_term_memory]
        return data


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_URL, timeout: int = 90) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def executable_available(self) -> bool:
        return shutil.which("ollama") is not None

    def server_available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=2) as response:
                return response.status == 200
        except Exception:  # noqa: BLE001
            return False

    def list_models(self) -> list[str]:
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return sorted(model.get("name", "") for model in payload.get("models", []) if model.get("name"))
        except Exception:  # noqa: BLE001
            return []

    def generate(self, model: str, prompt: str) -> str:
        data = json.dumps({"model": model, "prompt": prompt, "stream": False}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return str(payload.get("response", "")).strip()


class AgentEngine(threading.Thread):
    def __init__(
        self,
        state: AppState,
        ollama: OllamaClient,
        state_lock: threading.RLock,
        notify: Callable[[str], None],
        save_callback: Callable[[], None],
    ) -> None:
        super().__init__(daemon=True)
        self.state = state
        self.ollama = ollama
        self.state_lock = state_lock
        self.notify = notify
        self.save_callback = save_callback
        self.stop_event = threading.Event()
        self.processed_messages: dict[str, int] = {}

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                with self.state_lock:
                    paused = self.state.paused
                    server_ok = self.ollama.server_available()
                    agents = [agent for agent in self.state.agents if agent.active]
                if not paused and server_ok:
                    for agent in agents:
                        if self.stop_event.is_set():
                            break
                        self._tick_agent(agent.id)
                time.sleep(ENGINE_INTERVAL_SECONDS)
            except Exception as exc:  # noqa: BLE001 - engine must be resilient
                self._log_event(f"Ошибка движка: {exc}\n{traceback.format_exc(limit=3)}")
                time.sleep(ENGINE_INTERVAL_SECONDS)

    def shutdown(self) -> None:
        self.stop_event.set()

    def _tick_agent(self, agent_id: str) -> None:
        with self.state_lock:
            agent = self._find_agent(agent_id)
            if agent is None or not agent.active or self.state.paused:
                return
            model = agent.model or self.state.selected_global_model
            if not model:
                return
            message_count = len(self.state.messages)
            last_processed = self.processed_messages.get(agent.id, 0)
            new_messages = self.state.messages[last_processed:]
            visible_new = [m for m in new_messages if m.sender != agent.name and (m.recipient in ("Все", agent.name) or m.sender != "Пользователь")]
            should_initiate = message_count == last_processed and len(self.state.messages) > 0
            if not visible_new and not should_initiate:
                return
            prompt = self._build_prompt(agent, visible_new or self.state.messages[-3:])
        try:
            raw_response = self.ollama.generate(model, prompt)
            action = self._parse_agent_response(raw_response)
            with self.state_lock:
                self.processed_messages[agent_id] = len(self.state.messages)
                self._apply_action(agent_id, action, raw_response)
            self.save_callback()
            self.notify("state_changed")
        except Exception as exc:  # noqa: BLE001
            self._log_event(f"Агент {agent.name}: ошибка запроса к Ollama: {exc}")

    def _build_prompt(self, agent: Agent, new_messages: list[Message]) -> str:
        chat = "\n".join(
            f"[{message.created_at}] {message.sender} → {message.recipient}: {message.text}"
            for message in self.state.messages[-MAX_CHAT_CONTEXT:]
        )
        incoming = "\n".join(f"{m.sender} → {m.recipient}: {m.text}" for m in new_messages)
        long_memory = "\n".join(agent.long_term_memory[-20:]) or "Пока нет."
        short_memory = "\n".join(f"{m.sender}: {m.text}" for m in agent.short_term_memory[-agent.memory_limit :]) or "Пока нет."
        other_agents = ", ".join(a.name for a in self.state.agents if a.id != agent.id) or "нет"
        return f"""
Ты агент многоагентной локальной чат-системы.
Имя: {agent.name}
Роль: {agent.role}
Цель: {agent.goal}
Системный промпт: {agent.system_prompt}
Другие агенты: {other_agents}

Долгосрочная память:
{long_memory}

Краткосрочная память:
{short_memory}

Недавняя история чата:
{chat}

Новые входящие сообщения:
{incoming or "Нет новых сообщений."}

Реши, нужно ли ответить. Верни СТРОГО один JSON-объект без Markdown:
{{
  "respond": true или false,
  "message": "текст сообщения на русском языке или пустая строка",
  "recipient": "Все" или имя агента или "Пользователь",
  "memory": "важный вывод для долгосрочной памяти или пустая строка",
  "state_update": {{"goal": "новая цель или пустая строка", "system_prompt": "новый системный промпт или пустая строка"}},
  "create_agent": null или {{"name": "имя", "role": "роль", "system_prompt": "промпт", "goal": "цель", "memory_limit": 20}}
}}
Если отвечать не нужно, установи respond=false.
""".strip()

    def _parse_agent_response(self, raw_response: str) -> dict[str, Any]:
        text = raw_response.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return {"respond": True, "message": raw_response[:2000], "recipient": "Все", "memory": "", "state_update": {}, "create_agent": None}

    def _apply_action(self, agent_id: str, action: dict[str, Any], raw_response: str) -> None:
        agent = self._find_agent(agent_id)
        if agent is None:
            return
        memory = str(action.get("memory") or "").strip()
        if memory:
            agent.remember_event(memory)
            self.state.events.append(f"{now_iso()} — {agent.name} сохранил память: {memory}")
        state_update = action.get("state_update") if isinstance(action.get("state_update"), dict) else {}
        if state_update.get("goal"):
            agent.goal = str(state_update["goal"])
            self.state.events.append(f"{now_iso()} — {agent.name} изменил цель.")
        if state_update.get("system_prompt"):
            agent.system_prompt = str(state_update["system_prompt"])
            self.state.events.append(f"{now_iso()} — {agent.name} изменил системный промпт.")
        create_agent = action.get("create_agent")
        if isinstance(create_agent, dict) and create_agent.get("name"):
            new_agent = Agent(
                id=str(uuid.uuid4()),
                name=str(create_agent.get("name", "Новый агент"))[:80],
                role=str(create_agent.get("role", "Создан другим агентом"))[:300],
                system_prompt=str(create_agent.get("system_prompt", "Отвечай на русском языке."))[:4000],
                goal=str(create_agent.get("goal", "Участвовать в общей среде общения."))[:1000],
                model=agent.model,
                memory_limit=int(create_agent.get("memory_limit", MAX_SHORT_TERM_MEMORY) or MAX_SHORT_TERM_MEMORY),
            )
            self.state.agents.append(new_agent)
            self.state.events.append(f"{now_iso()} — {agent.name} создал агента {new_agent.name}.")
        if bool(action.get("respond")):
            message_text = str(action.get("message") or raw_response).strip()
            if message_text:
                recipient = str(action.get("recipient") or "Все").strip() or "Все"
                message = Message(sender=agent.name, recipient=recipient, text=message_text[:4000])
                self.state.messages.append(message)
                for item in self.state.agents:
                    if item.name in (agent.name, recipient) or recipient == "Все":
                        item.remember_message(message)
                self.state.events.append(f"{now_iso()} — {agent.name} отправил сообщение для {recipient}.")

    def _find_agent(self, agent_id: str) -> Agent | None:
        return next((agent for agent in self.state.agents if agent.id == agent_id), None)

    def _log_event(self, text: str) -> None:
        with self.state_lock:
            self.state.events.append(f"{now_iso()} — {text}")
        self.save_callback()
        self.notify("state_changed")


class AgentDialog(simpledialog.Dialog):
    def __init__(self, parent: tk.Misc, title: str, agent: Agent | None = None, models: list[str] | None = None) -> None:
        self.agent = agent
        self.models = models or []
        self.result_agent: Agent | None = None
        super().__init__(parent, title)

    def body(self, master: tk.Frame) -> tk.Widget:
        self.name_var = tk.StringVar(value=self.agent.name if self.agent else "")
        self.role_var = tk.StringVar(value=self.agent.role if self.agent else "")
        self.goal_var = tk.StringVar(value=self.agent.goal if self.agent else "")
        self.model_var = tk.StringVar(value=self.agent.model if self.agent else "")
        self.memory_var = tk.IntVar(value=self.agent.memory_limit if self.agent else MAX_SHORT_TERM_MEMORY)
        self.active_var = tk.BooleanVar(value=self.agent.active if self.agent else True)

        ttk.Label(master, text="Имя:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(master, textvariable=self.name_var, width=48).grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        ttk.Label(master, text="Роль:").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(master, textvariable=self.role_var, width=48).grid(row=1, column=1, sticky="ew", padx=4, pady=4)
        ttk.Label(master, text="Цель:").grid(row=2, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(master, textvariable=self.goal_var, width=48).grid(row=2, column=1, sticky="ew", padx=4, pady=4)
        ttk.Label(master, text="Модель:").grid(row=3, column=0, sticky="w", padx=4, pady=4)
        ttk.Combobox(master, textvariable=self.model_var, values=[""] + self.models, width=46).grid(row=3, column=1, sticky="ew", padx=4, pady=4)
        ttk.Label(master, text="Память, сообщений:").grid(row=4, column=0, sticky="w", padx=4, pady=4)
        ttk.Spinbox(master, textvariable=self.memory_var, from_=1, to=200, width=10).grid(row=4, column=1, sticky="w", padx=4, pady=4)
        ttk.Checkbutton(master, text="Активен", variable=self.active_var).grid(row=5, column=1, sticky="w", padx=4, pady=4)
        ttk.Label(master, text="Системный промпт:").grid(row=6, column=0, sticky="nw", padx=4, pady=4)
        self.prompt_text = tk.Text(master, width=58, height=8, wrap="word")
        self.prompt_text.grid(row=6, column=1, sticky="nsew", padx=4, pady=4)
        self.prompt_text.insert("1.0", self.agent.system_prompt if self.agent else "Отвечай на русском языке.")
        master.columnconfigure(1, weight=1)
        return master

    def validate(self) -> bool:
        if not self.name_var.get().strip():
            messagebox.showerror("Ошибка", "Введите имя агента.", parent=self)
            return False
        return True

    def apply(self) -> None:
        base = self.agent or Agent(id=str(uuid.uuid4()), name="", role="", system_prompt="", goal="")
        base.name = self.name_var.get().strip()
        base.role = self.role_var.get().strip()
        base.goal = self.goal_var.get().strip()
        base.model = self.model_var.get().strip()
        base.memory_limit = max(1, int(self.memory_var.get()))
        base.active = self.active_var.get()
        base.system_prompt = self.prompt_text.get("1.0", "end").strip()
        self.result_agent = base


class MultiAgentChatApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Локальная многоагентная чат-система")
        self.geometry("1180x760")
        self.minsize(980, 620)
        self.store = StateStore()
        self.state = self.store.load()
        self.lock = threading.RLock()
        self.ollama = OllamaClient()
        self.ui_queue: queue.Queue[str] = queue.Queue()
        self.models: list[str] = []
        self._build_ui()
        self.refresh_ollama_status()
        self.refresh_all()
        self.engine = AgentEngine(self.state, self.ollama, self.lock, self.ui_queue.put, self.save_state)
        self.engine.start()
        self.after(500, self._drain_ui_queue)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        root = ttk.PanedWindow(self, orient="horizontal")
        root.pack(fill="both", expand=True, padx=8, pady=8)

        chat_frame = ttk.Frame(root)
        root.add(chat_frame, weight=3)
        toolbar = ttk.Frame(chat_frame)
        toolbar.pack(fill="x", pady=(0, 6))
        self.pause_button = ttk.Button(toolbar, command=self.toggle_pause)
        self.pause_button.pack(side="left", padx=(0, 6))
        ttk.Button(toolbar, text="Сохранить", command=self.save_state).pack(side="left", padx=3)
        ttk.Button(toolbar, text="Очистить чат", command=self.clear_chat).pack(side="left", padx=3)
        self.status_var = tk.StringVar(value="Проверка Ollama...")
        ttk.Label(toolbar, textvariable=self.status_var).pack(side="right")

        self.chat_text = tk.Text(chat_frame, wrap="word", state="disabled", height=28)
        self.chat_text.pack(fill="both", expand=True)
        self.chat_text.tag_configure("Пользователь", foreground="#0057b8", font=("Segoe UI", 10, "bold"))
        self.chat_text.tag_configure("Система", foreground="#777777", font=("Segoe UI", 10, "italic"))
        self.chat_text.bind("<Control-c>", lambda event: None)

        compose = ttk.Frame(chat_frame)
        compose.pack(fill="x", pady=(6, 0))
        ttk.Label(compose, text="Адресат:").pack(side="left")
        self.recipient_var = tk.StringVar(value="Все")
        self.recipient_combo = ttk.Combobox(compose, textvariable=self.recipient_var, state="readonly", width=22)
        self.recipient_combo.pack(side="left", padx=6)
        self.entry = tk.Text(compose, height=4, wrap="word")
        self.entry.pack(side="left", fill="x", expand=True, padx=6)
        self.entry.bind("<Control-Return>", self.send_user_message)
        ttk.Button(compose, text="Отправить", command=self.send_user_message).pack(side="right")

        side = ttk.Notebook(root)
        root.add(side, weight=1)
        agents_tab = ttk.Frame(side)
        settings_tab = ttk.Frame(side)
        events_tab = ttk.Frame(side)
        side.add(agents_tab, text="Агенты")
        side.add(settings_tab, text="Ollama и модели")
        side.add(events_tab, text="События")

        self.agent_list = tk.Listbox(agents_tab, height=18)
        self.agent_list.pack(fill="both", expand=True, padx=6, pady=6)
        buttons = ttk.Frame(agents_tab)
        buttons.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Button(buttons, text="Создать", command=self.create_agent).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(buttons, text="Редактировать", command=self.edit_agent).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(buttons, text="Удалить", command=self.delete_agent).pack(side="left", expand=True, fill="x", padx=2)

        ttk.Label(settings_tab, text="Глобальная модель:").pack(anchor="w", padx=6, pady=(8, 2))
        self.model_var = tk.StringVar(value=self.state.selected_global_model)
        self.model_combo = ttk.Combobox(settings_tab, textvariable=self.model_var, state="readonly")
        self.model_combo.pack(fill="x", padx=6, pady=2)
        ttk.Button(settings_tab, text="Обновить список моделей", command=self.refresh_ollama_status).pack(fill="x", padx=6, pady=4)
        ttk.Button(settings_tab, text="Открыть установку Ollama", command=self.open_ollama_download).pack(fill="x", padx=6, pady=4)
        ttk.Button(settings_tab, text="Запустить ollama serve", command=self.start_ollama_server).pack(fill="x", padx=6, pady=4)
        ttk.Button(settings_tab, text="Применить модель", command=self.apply_global_model).pack(fill="x", padx=6, pady=4)
        self.ollama_details = tk.Text(settings_tab, height=12, wrap="word", state="disabled")
        self.ollama_details.pack(fill="both", expand=True, padx=6, pady=6)

        self.events_text = tk.Text(events_tab, wrap="word", state="disabled")
        self.events_text.pack(fill="both", expand=True, padx=6, pady=6)

    def refresh_all(self) -> None:
        self.refresh_chat()
        self.refresh_agents()
        self.refresh_events()
        self.pause_button.configure(text="Возобновить" if self.state.paused else "Пауза")

    def refresh_chat(self) -> None:
        self.chat_text.configure(state="normal")
        self.chat_text.delete("1.0", "end")
        for message in self.state.messages[-500:]:
            header = f"[{message.created_at}] {message.sender} → {message.recipient}\n"
            tag = message.sender if message.sender in ("Пользователь", "Система") else "agent"
            if tag == "agent" and not self.chat_text.tag_names().__contains__(message.sender):
                self.chat_text.tag_configure(message.sender, foreground="#188038", font=("Segoe UI", 10, "bold"))
                tag = message.sender
            self.chat_text.insert("end", header, tag)
            self.chat_text.insert("end", f"{message.text}\n\n")
        self.chat_text.configure(state="disabled")
        self.chat_text.see("end")

    def refresh_agents(self) -> None:
        self.agent_list.delete(0, "end")
        recipients = ["Все"]
        for agent in self.state.agents:
            mark = "●" if agent.active else "○"
            model = f" [{agent.model}]" if agent.model else ""
            self.agent_list.insert("end", f"{mark} {agent.name} — {agent.role}{model}")
            recipients.append(agent.name)
        self.recipient_combo.configure(values=recipients)
        if self.recipient_var.get() not in recipients:
            self.recipient_var.set("Все")

    def refresh_events(self) -> None:
        self.events_text.configure(state="normal")
        self.events_text.delete("1.0", "end")
        self.events_text.insert("end", "\n".join(self.state.events[-1000:]))
        self.events_text.configure(state="disabled")
        self.events_text.see("end")

    def refresh_ollama_status(self) -> None:
        executable = self.ollama.executable_available()
        server = self.ollama.server_available()
        self.models = self.ollama.list_models() if server else []
        self.model_combo.configure(values=self.models)
        if self.models and not self.model_var.get():
            self.model_var.set(self.models[0])
        if server:
            self.status_var.set(f"Ollama доступна, моделей: {len(self.models)}")
        elif executable:
            self.status_var.set("Ollama установлена, но сервер не отвечает")
        else:
            self.status_var.set("Ollama не найдена")
            messagebox.showwarning(
                "Ollama не найдена",
                "Ollama не установлена или недоступна. Установите Ollama и загрузите локальную модель, например: ollama pull llama3.1",
                parent=self,
            )
        details = [
            f"Исполняемый файл Ollama: {'найден' if executable else 'не найден'}",
            f"Локальный сервер {OLLAMA_URL}: {'доступен' if server else 'недоступен'}",
            "Доступные модели:",
            *(f" • {model}" for model in self.models),
        ]
        self.ollama_details.configure(state="normal")
        self.ollama_details.delete("1.0", "end")
        self.ollama_details.insert("end", "\n".join(details))
        self.ollama_details.configure(state="disabled")
        self.refresh_agents()

    def send_user_message(self, event: tk.Event | None = None) -> str:
        text = self.entry.get("1.0", "end").strip()
        if not text:
            return "break"
        with self.lock:
            message = Message(sender="Пользователь", recipient=self.recipient_var.get(), text=text)
            self.state.messages.append(message)
            for agent in self.state.agents:
                if message.recipient in ("Все", agent.name):
                    agent.remember_message(message)
            self.state.events.append(f"{now_iso()} — Пользователь отправил сообщение для {message.recipient}.")
        self.entry.delete("1.0", "end")
        self.save_state()
        self.refresh_all()
        return "break"

    def toggle_pause(self) -> None:
        with self.lock:
            self.state.paused = not self.state.paused
            self.state.events.append(f"{now_iso()} — Симуляция {'поставлена на паузу' if self.state.paused else 'возобновлена'}.")
        self.save_state()
        self.refresh_all()

    def clear_chat(self) -> None:
        if messagebox.askyesno("Очистить чат", "Удалить историю сообщений из интерфейса? Память агентов останется.", parent=self):
            with self.lock:
                self.state.messages.clear()
                self.state.events.append(f"{now_iso()} — История чата очищена пользователем.")
            self.save_state()
            self.refresh_all()

    def create_agent(self) -> None:
        dialog = AgentDialog(self, "Создание агента", models=self.models)
        if dialog.result_agent:
            with self.lock:
                self.state.agents.append(dialog.result_agent)
                self.state.events.append(f"{now_iso()} — Создан агент {dialog.result_agent.name}.")
            self.save_state()
            self.refresh_all()

    def edit_agent(self) -> None:
        agent = self._selected_agent()
        if not agent:
            messagebox.showinfo("Агент", "Выберите агента в списке.", parent=self)
            return
        dialog = AgentDialog(self, "Редактирование агента", agent=agent, models=self.models)
        if dialog.result_agent:
            with self.lock:
                self.state.events.append(f"{now_iso()} — Изменены параметры агента {dialog.result_agent.name}.")
            self.save_state()
            self.refresh_all()

    def delete_agent(self) -> None:
        agent = self._selected_agent()
        if not agent:
            messagebox.showinfo("Агент", "Выберите агента в списке.", parent=self)
            return
        if messagebox.askyesno("Удаление агента", f"Удалить агента {agent.name}?", parent=self):
            with self.lock:
                self.state.agents = [item for item in self.state.agents if item.id != agent.id]
                self.state.events.append(f"{now_iso()} — Удалён агент {agent.name}.")
            self.save_state()
            self.refresh_all()

    def _selected_agent(self) -> Agent | None:
        selection = self.agent_list.curselection()
        if not selection:
            return None
        index = selection[0]
        return self.state.agents[index] if index < len(self.state.agents) else None

    def apply_global_model(self) -> None:
        with self.lock:
            self.state.selected_global_model = self.model_var.get()
            self.state.events.append(f"{now_iso()} — Выбрана глобальная модель: {self.state.selected_global_model or 'не задана'}.")
        self.save_state()
        self.refresh_all()

    def open_ollama_download(self) -> None:
        webbrowser.open(OLLAMA_DOWNLOAD_URL)

    def start_ollama_server(self) -> None:
        if not self.ollama.executable_available():
            messagebox.showwarning("Ollama", "Исполняемый файл Ollama не найден. Сначала установите Ollama.", parent=self)
            return
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.state.events.append(f"{now_iso()} — Выполнена попытка запуска ollama serve.")
            self.after(1500, self.refresh_ollama_status)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Ollama", f"Не удалось запустить Ollama: {exc}", parent=self)

    def save_state(self) -> None:
        with self.lock:
            self.store.save(self.state)

    def _drain_ui_queue(self) -> None:
        changed = False
        while True:
            try:
                self.ui_queue.get_nowait()
                changed = True
            except queue.Empty:
                break
        if changed:
            self.refresh_all()
        self.after(500, self._drain_ui_queue)

    def on_close(self) -> None:
        self.engine.shutdown()
        self.save_state()
        self.destroy()


def main() -> None:
    if sys.platform != "win32":
        print("Приложение ориентировано на Windows, но может запускаться на других ОС при наличии Tkinter и Ollama.")
    app = MultiAgentChatApp()
    app.mainloop()


if __name__ == "__main__":
    main()
