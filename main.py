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
from typing import Any, Callable, Protocol
import base64
import hashlib

import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_DOWNLOAD_URL = "https://ollama.com/download/windows"
ENGINE_INTERVAL_SECONDS = 5
MAX_CHAT_CONTEXT = 30
MAX_SHORT_TERM_MEMORY = 20
DEFAULT_PROVIDER_MODELS = {
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
    "anthropic": ["claude-3-5-haiku-latest", "claude-3-5-sonnet-latest"],
    "gemini": ["gemini-1.5-flash", "gemini-1.5-pro"],
    "openrouter": ["openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet", "google/gemini-flash-1.5"],
}
PROVIDER_NAMES = {
    "ollama": "Ollama",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "gemini": "Google Gemini",
    "openrouter": "OpenRouter",
}


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
class ProviderConfig:
    id: str
    name: str
    provider_type: str
    encrypted_api_key: str = ""
    enabled: bool = True
    base_url: str = ""
    default_model: str = ""
    models: list[str] = field(default_factory=list)

    def masked_key(self) -> str:
        if not self.encrypted_api_key:
            return "ключ не задан"
        return "••••••••"


@dataclass
class ProviderStatus:
    provider_id: str
    available: bool = False
    status: str = "не проверялся"
    active_model: str = ""
    last_latency_ms: int = 0
    request_count: int = 0
    last_error: str = ""


@dataclass(frozen=True)
class ModelSelection:
    provider_id: str
    model: str

    @classmethod
    def from_string(cls, value: str, default_provider: str = "ollama") -> "ModelSelection":
        if ":" in value:
            provider_id, model = value.split(":", 1)
            return cls(provider_id.strip() or default_provider, model.strip())
        return cls(default_provider, value.strip())

    def label(self) -> str:
        return f"{self.provider_id}:{self.model}" if self.model else self.provider_id


class LocalKeyCipher:
    """Small local reversible cipher for API keys without third-party packages."""

    def __init__(self, secret_path: Path | None = None) -> None:
        self.secret_path = secret_path or (app_data_dir() / ".key_secret")
        self.secret_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.secret_path.exists():
            self.secret_path.write_text(uuid.uuid4().hex + uuid.uuid4().hex, encoding="utf-8")
        self.secret = self.secret_path.read_text(encoding="utf-8").strip().encode("utf-8")

    def _keystream(self, length: int) -> bytes:
        chunks: list[bytes] = []
        counter = 0
        while sum(len(chunk) for chunk in chunks) < length:
            chunks.append(hashlib.sha256(self.secret + counter.to_bytes(4, "big")).digest())
            counter += 1
        return b"".join(chunks)[:length]

    def encrypt(self, value: str) -> str:
        raw = value.encode("utf-8")
        stream = self._keystream(len(raw))
        encrypted = bytes(byte ^ stream[index] for index, byte in enumerate(raw))
        return base64.urlsafe_b64encode(encrypted).decode("ascii")

    def decrypt(self, value: str) -> str:
        try:
            raw = base64.urlsafe_b64decode(value.encode("ascii"))
            stream = self._keystream(len(raw))
            decrypted = bytes(byte ^ stream[index] for index, byte in enumerate(raw))
            return decrypted.decode("utf-8")
        except Exception:
            return ""


@dataclass
class Agent:
    id: str
    name: str
    role: str
    system_prompt: str
    goal: str
    model: str = ""
    provider_id: str = "ollama"
    fallback_provider_id: str = ""
    fallback_model: str = ""
    active: bool = True
    memory_limit: int = 20
    memory_settings: dict[str, Any] = field(default_factory=dict)
    activity_settings: dict[str, Any] = field(default_factory=dict)
    behavior_params: dict[str, Any] = field(default_factory=dict)
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
    selected_global_provider: str = "ollama"
    providers: list[ProviderConfig] = field(default_factory=list)
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
                selected_global_provider=raw.get("selected_global_provider", "ollama"),
                providers=[self._provider_from_dict(item) for item in raw.get("providers", [])] or self._default_providers(),
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
            "selected_global_provider": state.selected_global_provider,
            "providers": [asdict(provider) for provider in state.providers],
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
            messages=[Message(sender="Система", text="Добро пожаловать! Установите/запустите Ollama или добавьте API-ключ облачного провайдера, выберите модель и снимите систему с паузы.", kind="system")],
            events=[f"{now_iso()} — Создано начальное состояние."],
            providers=self._default_providers(),
            paused=True,
        )

    def _default_providers(self) -> list[ProviderConfig]:
        return [
            ProviderConfig(id="ollama", name="Ollama", provider_type="ollama", enabled=True),
            ProviderConfig(id="openai", name="OpenAI", provider_type="openai", models=DEFAULT_PROVIDER_MODELS["openai"]),
            ProviderConfig(id="anthropic", name="Anthropic", provider_type="anthropic", models=DEFAULT_PROVIDER_MODELS["anthropic"]),
            ProviderConfig(id="gemini", name="Google Gemini", provider_type="gemini", models=DEFAULT_PROVIDER_MODELS["gemini"]),
            ProviderConfig(id="openrouter", name="OpenRouter", provider_type="openrouter", models=DEFAULT_PROVIDER_MODELS["openrouter"]),
        ]

    def _provider_from_dict(self, data: dict[str, Any]) -> ProviderConfig:
        provider_type = data.get("provider_type", data.get("id", "openai"))
        return ProviderConfig(
            id=data.get("id", provider_type),
            name=data.get("name", PROVIDER_NAMES.get(provider_type, provider_type)),
            provider_type=provider_type,
            encrypted_api_key=data.get("encrypted_api_key", ""),
            enabled=bool(data.get("enabled", True)),
            base_url=data.get("base_url", ""),
            default_model=data.get("default_model", ""),
            models=list(data.get("models", [])) or DEFAULT_PROVIDER_MODELS.get(provider_type, []),
        )

    def _agent_from_dict(self, data: dict[str, Any]) -> Agent:
        return Agent(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", "Новый агент"),
            role=data.get("role", ""),
            system_prompt=data.get("system_prompt", ""),
            goal=data.get("goal", ""),
            model=data.get("model", ""),
            provider_id=data.get("provider_id", "ollama"),
            fallback_provider_id=data.get("fallback_provider_id", ""),
            fallback_model=data.get("fallback_model", ""),
            active=bool(data.get("active", True)),
            memory_limit=int(data.get("memory_limit", MAX_SHORT_TERM_MEMORY) or MAX_SHORT_TERM_MEMORY),
            memory_settings=dict(data.get("memory_settings", {})),
            activity_settings=dict(data.get("activity_settings", {})),
            behavior_params=dict(data.get("behavior_params", {})),
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




class ModelProvider(Protocol):
    def list_models(self) -> list[str]: ...
    def generate(self, model: str, prompt: str) -> str: ...
    def test_connection(self, model: str = "") -> tuple[bool, str, int]: ...


class OllamaProvider:
    def __init__(self, client: OllamaClient) -> None:
        self.client = client

    def list_models(self) -> list[str]:
        return self.client.list_models()

    def generate(self, model: str, prompt: str) -> str:
        return self.client.generate(model, prompt)

    def test_connection(self, model: str = "") -> tuple[bool, str, int]:
        started = time.perf_counter()
        if not self.client.executable_available():
            return False, "исполняемый файл Ollama не найден", 0
        if not self.client.server_available():
            return False, "локальный сервер Ollama не отвечает", 0
        latency = int((time.perf_counter() - started) * 1000)
        return True, "Ollama доступна", latency


class CloudProvider:
    def __init__(self, config: ProviderConfig, cipher: LocalKeyCipher, timeout: int = 90) -> None:
        self.config = config
        self.cipher = cipher
        self.timeout = timeout

    @property
    def api_key(self) -> str:
        return self.cipher.decrypt(self.config.encrypted_api_key)

    def list_models(self) -> list[str]:
        return list(self.config.models or DEFAULT_PROVIDER_MODELS.get(self.config.provider_type, []))

    def generate(self, model: str, prompt: str) -> str:
        if not self.config.enabled:
            raise RuntimeError("провайдер временно отключён")
        if not self.api_key:
            raise RuntimeError("API-ключ не задан")
        provider_type = self.config.provider_type
        if provider_type in ("openai", "openrouter"):
            return self._openai_compatible_generate(model, prompt)
        if provider_type == "anthropic":
            return self._anthropic_generate(model, prompt)
        if provider_type == "gemini":
            return self._gemini_generate(model, prompt)
        raise RuntimeError(f"неизвестный провайдер: {provider_type}")

    def test_connection(self, model: str = "") -> tuple[bool, str, int]:
        started = time.perf_counter()
        try:
            selected_model = model or self.config.default_model or (self.list_models()[0] if self.list_models() else "")
            if not selected_model:
                return False, "модель не задана", 0
            self.generate(selected_model, "Ответь одним словом: OK")
            return True, "подключение успешно", int((time.perf_counter() - started) * 1000)
        except Exception as exc:  # noqa: BLE001
            return False, str(exc), int((time.perf_counter() - started) * 1000)

    def _request_json(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"HTTP {exc.code}: {body}") from exc

    def _openai_compatible_generate(self, model: str, prompt: str) -> str:
        base_url = self.config.base_url.rstrip("/") or (
            "https://openrouter.ai/api/v1" if self.config.provider_type == "openrouter" else "https://api.openai.com/v1"
        )
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self.config.provider_type == "openrouter":
            headers.update({"HTTP-Referer": "http://localhost", "X-Title": "LocalMultiAgentChat"})
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7}
        data = self._request_json(f"{base_url}/chat/completions", payload, headers)
        return str(data.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()

    def _anthropic_generate(self, model: str, prompt: str) -> str:
        base_url = self.config.base_url.rstrip("/") or "https://api.anthropic.com"
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        payload = {"model": model, "max_tokens": 2048, "messages": [{"role": "user", "content": prompt}]}
        data = self._request_json(f"{base_url}/v1/messages", payload, headers)
        parts = data.get("content", [])
        return "\n".join(str(part.get("text", "")) for part in parts if isinstance(part, dict)).strip()

    def _gemini_generate(self, model: str, prompt: str) -> str:
        base_url = self.config.base_url.rstrip("/") or "https://generativelanguage.googleapis.com/v1beta"
        encoded_model = urllib.parse.quote(model, safe="")
        url = f"{base_url}/models/{encoded_model}:generateContent?key={urllib.parse.quote(self.api_key)}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        data = self._request_json(url, payload, {})
        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return "\n".join(str(part.get("text", "")) for part in parts if isinstance(part, dict)).strip()


class ModelManager:
    def __init__(self, state: AppState, ollama: OllamaClient, cipher: LocalKeyCipher | None = None) -> None:
        self.state = state
        self.ollama = ollama
        self.cipher = cipher or LocalKeyCipher()
        self.statuses: dict[str, ProviderStatus] = {}

    def provider_config(self, provider_id: str) -> ProviderConfig | None:
        return next((provider for provider in self.state.providers if provider.id == provider_id), None)

    def provider(self, provider_id: str) -> ModelProvider:
        if provider_id == "ollama":
            return OllamaProvider(self.ollama)
        config = self.provider_config(provider_id)
        if config is None:
            raise RuntimeError(f"провайдер {provider_id} не найден")
        return CloudProvider(config, self.cipher)

    def list_models(self, provider_id: str) -> list[str]:
        try:
            models = self.provider(provider_id).list_models()
            if provider_id != "ollama":
                config = self.provider_config(provider_id)
                if config is not None and models:
                    config.models = models
            return models
        except Exception:
            return []

    def all_model_labels(self) -> list[str]:
        labels: list[str] = []
        for provider in self.state.providers:
            for model in self.list_models(provider.id):
                labels.append(ModelSelection(provider.id, model).label())
        return sorted(labels)

    def generate(self, primary: ModelSelection, prompt: str, fallback: ModelSelection | None = None) -> tuple[str, ModelSelection]:
        try:
            return self._generate_once(primary, prompt), primary
        except Exception as exc:  # noqa: BLE001
            self._mark_error(primary.provider_id, str(exc), primary.model)
            if fallback and fallback.provider_id and fallback.model and fallback != primary:
                return self._generate_once(fallback, prompt), fallback
            raise

    def test_connection(self, provider_id: str, model: str = "") -> tuple[bool, str, int]:
        ok, message, latency = self.provider(provider_id).test_connection(model)
        status = self.statuses.setdefault(provider_id, ProviderStatus(provider_id=provider_id))
        status.available = ok
        status.status = message
        status.last_latency_ms = latency
        status.last_error = "" if ok else message
        status.active_model = model or status.active_model
        return ok, message, latency

    def _generate_once(self, selection: ModelSelection, prompt: str) -> str:
        started = time.perf_counter()
        text = self.provider(selection.provider_id).generate(selection.model, prompt)
        latency = int((time.perf_counter() - started) * 1000)
        status = self.statuses.setdefault(selection.provider_id, ProviderStatus(provider_id=selection.provider_id))
        status.available = True
        status.status = "доступен"
        status.active_model = selection.model
        status.last_latency_ms = latency
        status.request_count += 1
        status.last_error = ""
        return text

    def _mark_error(self, provider_id: str, error: str, model: str = "") -> None:
        status = self.statuses.setdefault(provider_id, ProviderStatus(provider_id=provider_id))
        status.available = False
        status.status = "ошибка"
        status.active_model = model
        status.last_error = error


class AgentEngine(threading.Thread):
    def __init__(
        self,
        state: AppState,
        model_manager: ModelManager,
        state_lock: threading.RLock,
        notify: Callable[[str], None],
        save_callback: Callable[[], None],
    ) -> None:
        super().__init__(daemon=True)
        self.state = state
        self.model_manager = model_manager
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
                    agents = [agent for agent in self.state.agents if agent.active]
                if not paused:
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
            provider_id = agent.provider_id or self.state.selected_global_provider or "ollama"
            model = agent.model or self.state.selected_global_model
            if not model:
                return
            primary = ModelSelection(provider_id, model)
            fallback = None
            if agent.fallback_provider_id and agent.fallback_model:
                fallback = ModelSelection(agent.fallback_provider_id, agent.fallback_model)
            message_count = len(self.state.messages)
            last_processed = self.processed_messages.get(agent.id, 0)
            new_messages = self.state.messages[last_processed:]
            visible_new = [m for m in new_messages if m.sender != agent.name and (m.recipient in ("Все", agent.name) or m.sender != "Пользователь")]
            should_initiate = message_count == last_processed and len(self.state.messages) > 0
            if not visible_new and not should_initiate:
                return
            prompt = self._build_prompt(agent, visible_new or self.state.messages[-3:])
        try:
            raw_response, used_model = self.model_manager.generate(primary, prompt, fallback)
            action = self._parse_agent_response(raw_response)
            with self.state_lock:
                self.processed_messages[agent_id] = len(self.state.messages)
                if used_model != primary:
                    self.state.events.append(f"{now_iso()} — {agent.name}: использована резервная модель {used_model.label()}.")
                self._apply_action(agent_id, action, raw_response)
            self.save_callback()
            self.notify("state_changed")
        except Exception as exc:  # noqa: BLE001
            self._log_event(f"Агент {agent.name}: ошибка запроса к менеджеру моделей: {exc}")

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
                provider_id=agent.provider_id,
                fallback_provider_id=agent.fallback_provider_id,
                fallback_model=agent.fallback_model,
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
    def __init__(self, parent: tk.Misc, title: str, agent: Agent | None = None, models: list[str] | None = None, providers: list[ProviderConfig] | None = None) -> None:
        self.agent = agent
        self.models = models or []
        self.providers = providers or []
        self.result_agent: Agent | None = None
        super().__init__(parent, title)

    def body(self, master: tk.Frame) -> tk.Widget:
        self.name_var = tk.StringVar(value=self.agent.name if self.agent else "")
        self.role_var = tk.StringVar(value=self.agent.role if self.agent else "")
        self.goal_var = tk.StringVar(value=self.agent.goal if self.agent else "")
        self.provider_var = tk.StringVar(value=self.agent.provider_id if self.agent else "ollama")
        self.model_var = tk.StringVar(value=self.agent.model if self.agent else "")
        self.fallback_provider_var = tk.StringVar(value=self.agent.fallback_provider_id if self.agent else "")
        self.fallback_model_var = tk.StringVar(value=self.agent.fallback_model if self.agent else "")
        self.memory_var = tk.IntVar(value=self.agent.memory_limit if self.agent else MAX_SHORT_TERM_MEMORY)
        self.active_var = tk.BooleanVar(value=self.agent.active if self.agent else True)
        self.memory_settings_var = tk.StringVar(value=json.dumps(self.agent.memory_settings, ensure_ascii=False) if self.agent and self.agent.memory_settings else "{}")
        self.activity_settings_var = tk.StringVar(value=json.dumps(self.agent.activity_settings, ensure_ascii=False) if self.agent and self.agent.activity_settings else "{}")
        self.behavior_params_var = tk.StringVar(value=json.dumps(self.agent.behavior_params, ensure_ascii=False) if self.agent and self.agent.behavior_params else "{}")

        ttk.Label(master, text="Имя:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(master, textvariable=self.name_var, width=48).grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        ttk.Label(master, text="Роль:").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(master, textvariable=self.role_var, width=48).grid(row=1, column=1, sticky="ew", padx=4, pady=4)
        ttk.Label(master, text="Цель:").grid(row=2, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(master, textvariable=self.goal_var, width=48).grid(row=2, column=1, sticky="ew", padx=4, pady=4)
        provider_values = [provider.id for provider in self.providers]
        ttk.Label(master, text="Провайдер:").grid(row=3, column=0, sticky="w", padx=4, pady=4)
        ttk.Combobox(master, textvariable=self.provider_var, values=provider_values, width=46).grid(row=3, column=1, sticky="ew", padx=4, pady=4)
        ttk.Label(master, text="Модель:").grid(row=4, column=0, sticky="w", padx=4, pady=4)
        ttk.Combobox(master, textvariable=self.model_var, values=[""] + self.models, width=46).grid(row=4, column=1, sticky="ew", padx=4, pady=4)
        ttk.Label(master, text="Резервный провайдер:").grid(row=5, column=0, sticky="w", padx=4, pady=4)
        ttk.Combobox(master, textvariable=self.fallback_provider_var, values=[""] + provider_values, width=46).grid(row=5, column=1, sticky="ew", padx=4, pady=4)
        ttk.Label(master, text="Резервная модель:").grid(row=6, column=0, sticky="w", padx=4, pady=4)
        ttk.Combobox(master, textvariable=self.fallback_model_var, values=[""] + self.models, width=46).grid(row=6, column=1, sticky="ew", padx=4, pady=4)
        ttk.Label(master, text="Память, сообщений:").grid(row=7, column=0, sticky="w", padx=4, pady=4)
        ttk.Spinbox(master, textvariable=self.memory_var, from_=1, to=200, width=10).grid(row=7, column=1, sticky="w", padx=4, pady=4)
        ttk.Checkbutton(master, text="Активен", variable=self.active_var).grid(row=8, column=1, sticky="w", padx=4, pady=4)
        ttk.Label(master, text="Настройки памяти JSON:").grid(row=9, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(master, textvariable=self.memory_settings_var, width=48).grid(row=9, column=1, sticky="ew", padx=4, pady=4)
        ttk.Label(master, text="Настройки активности JSON:").grid(row=10, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(master, textvariable=self.activity_settings_var, width=48).grid(row=10, column=1, sticky="ew", padx=4, pady=4)
        ttk.Label(master, text="Параметры поведения JSON:").grid(row=11, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(master, textvariable=self.behavior_params_var, width=48).grid(row=11, column=1, sticky="ew", padx=4, pady=4)
        ttk.Label(master, text="Системный промпт:").grid(row=12, column=0, sticky="nw", padx=4, pady=4)
        self.prompt_text = tk.Text(master, width=58, height=8, wrap="word")
        self.prompt_text.grid(row=12, column=1, sticky="nsew", padx=4, pady=4)
        self.prompt_text.insert("1.0", self.agent.system_prompt if self.agent else "Отвечай на русском языке.")
        master.columnconfigure(1, weight=1)
        return master

    def validate(self) -> bool:
        if not self.name_var.get().strip():
            messagebox.showerror("Ошибка", "Введите имя агента.", parent=self)
            return False
        for label, value in (
            ("Настройки памяти", self.memory_settings_var.get()),
            ("Настройки активности", self.activity_settings_var.get()),
            ("Параметры поведения", self.behavior_params_var.get()),
        ):
            try:
                parsed = json.loads(value or "{}")
            except json.JSONDecodeError:
                messagebox.showerror("Ошибка", f"{label} должны быть корректным JSON-объектом.", parent=self)
                return False
            if not isinstance(parsed, dict):
                messagebox.showerror("Ошибка", f"{label} должны быть JSON-объектом.", parent=self)
                return False
        return True

    def apply(self) -> None:
        base = self.agent or Agent(id=str(uuid.uuid4()), name="", role="", system_prompt="", goal="")
        base.name = self.name_var.get().strip()
        base.role = self.role_var.get().strip()
        base.goal = self.goal_var.get().strip()
        base.provider_id = self.provider_var.get().strip() or "ollama"
        base.model = self.model_var.get().strip()
        if ":" in base.model:
            selection = ModelSelection.from_string(base.model)
            base.provider_id = selection.provider_id
            base.model = selection.model
        base.fallback_provider_id = self.fallback_provider_var.get().strip()
        base.fallback_model = self.fallback_model_var.get().strip()
        if ":" in base.fallback_model:
            fallback_selection = ModelSelection.from_string(base.fallback_model)
            base.fallback_provider_id = fallback_selection.provider_id
            base.fallback_model = fallback_selection.model
        base.memory_limit = max(1, int(self.memory_var.get()))
        base.memory_settings = json.loads(self.memory_settings_var.get() or "{}")
        base.activity_settings = json.loads(self.activity_settings_var.get() or "{}")
        base.behavior_params = json.loads(self.behavior_params_var.get() or "{}")
        base.active = self.active_var.get()
        base.system_prompt = self.prompt_text.get("1.0", "end").strip()
        self.result_agent = base

class ProviderDialog(simpledialog.Dialog):
    def __init__(self, parent: tk.Misc, title: str, cipher: LocalKeyCipher, provider: ProviderConfig | None = None) -> None:
        self.provider = provider
        self.cipher = cipher
        self.result_provider: ProviderConfig | None = None
        super().__init__(parent, title)

    def body(self, master: tk.Frame) -> tk.Widget:
        self.id_var = tk.StringVar(value=self.provider.id if self.provider else "")
        self.name_var = tk.StringVar(value=self.provider.name if self.provider else "")
        self.type_var = tk.StringVar(value=self.provider.provider_type if self.provider else "openai")
        self.enabled_var = tk.BooleanVar(value=self.provider.enabled if self.provider else True)
        self.base_url_var = tk.StringVar(value=self.provider.base_url if self.provider else "")
        self.default_model_var = tk.StringVar(value=self.provider.default_model if self.provider else "")
        self.models_var = tk.StringVar(value=", ".join(self.provider.models) if self.provider else "")
        self.key_var = tk.StringVar(value="")

        ttk.Label(master, text="ID провайдера:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(master, textvariable=self.id_var, width=44).grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        ttk.Label(master, text="Название:").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(master, textvariable=self.name_var, width=44).grid(row=1, column=1, sticky="ew", padx=4, pady=4)
        ttk.Label(master, text="Тип:").grid(row=2, column=0, sticky="w", padx=4, pady=4)
        ttk.Combobox(master, textvariable=self.type_var, values=["openai", "anthropic", "gemini", "openrouter"], width=42).grid(row=2, column=1, sticky="ew", padx=4, pady=4)
        ttk.Checkbutton(master, text="Включён", variable=self.enabled_var).grid(row=3, column=1, sticky="w", padx=4, pady=4)
        ttk.Label(master, text="API-ключ:").grid(row=4, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(master, textvariable=self.key_var, width=44, show="*").grid(row=4, column=1, sticky="ew", padx=4, pady=4)
        ttk.Label(master, text="Оставьте пустым, чтобы сохранить текущий ключ.").grid(row=5, column=1, sticky="w", padx=4)
        ttk.Label(master, text="Base URL:").grid(row=6, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(master, textvariable=self.base_url_var, width=44).grid(row=6, column=1, sticky="ew", padx=4, pady=4)
        ttk.Label(master, text="Модель по умолчанию:").grid(row=7, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(master, textvariable=self.default_model_var, width=44).grid(row=7, column=1, sticky="ew", padx=4, pady=4)
        ttk.Label(master, text="Модели (через запятую):").grid(row=8, column=0, sticky="nw", padx=4, pady=4)
        ttk.Entry(master, textvariable=self.models_var, width=44).grid(row=8, column=1, sticky="ew", padx=4, pady=4)
        master.columnconfigure(1, weight=1)
        return master

    def validate(self) -> bool:
        if not self.id_var.get().strip():
            messagebox.showerror("Ошибка", "Введите ID провайдера.", parent=self)
            return False
        provider_type = self.type_var.get().strip()
        if provider_type not in {"openai", "anthropic", "gemini", "openrouter"}:
            messagebox.showerror("Ошибка", "Выберите поддерживаемый тип провайдера.", parent=self)
            return False
        key = self.key_var.get().strip()
        if key and not self._looks_like_api_key(provider_type, key):
            messagebox.showerror("Ошибка", "API-ключ не похож на ключ выбранного провайдера.", parent=self)
            return False
        return True

    def _looks_like_api_key(self, provider_type: str, key: str) -> bool:
        if len(key) < 12:
            return False
        prefixes = {"openai": "sk-", "anthropic": "sk-ant-", "openrouter": "sk-or-"}
        expected = prefixes.get(provider_type)
        return key.startswith(expected) if expected else True

    def apply(self) -> None:
        provider_type = self.type_var.get().strip()
        provider = self.provider or ProviderConfig(id="", name="", provider_type=provider_type)
        provider.id = self.id_var.get().strip()
        provider.name = self.name_var.get().strip() or PROVIDER_NAMES.get(provider_type, provider_type)
        provider.provider_type = provider_type
        provider.enabled = self.enabled_var.get()
        provider.base_url = self.base_url_var.get().strip()
        provider.default_model = self.default_model_var.get().strip()
        provider.models = [item.strip() for item in self.models_var.get().split(",") if item.strip()] or DEFAULT_PROVIDER_MODELS.get(provider_type, [])
        if self.key_var.get().strip():
            provider.encrypted_api_key = self.cipher.encrypt(self.key_var.get().strip())
        self.result_provider = provider


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
        self.cipher = LocalKeyCipher()
        self.model_manager = ModelManager(self.state, self.ollama, self.cipher)
        self.ui_queue: queue.Queue[str] = queue.Queue()
        self.models: list[str] = []
        self._build_ui()
        self.refresh_model_status()
        self.refresh_all()
        self.engine = AgentEngine(self.state, self.model_manager, self.lock, self.ui_queue.put, self.save_state)
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
        self.status_var = tk.StringVar(value="Проверка провайдеров моделей...")
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
        side.add(settings_tab, text="Модели")
        side.add(events_tab, text="События")

        self.agent_list = tk.Listbox(agents_tab, height=18)
        self.agent_list.pack(fill="both", expand=True, padx=6, pady=6)
        buttons = ttk.Frame(agents_tab)
        buttons.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Button(buttons, text="Создать", command=self.create_agent).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(buttons, text="Редактировать", command=self.edit_agent).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(buttons, text="Удалить", command=self.delete_agent).pack(side="left", expand=True, fill="x", padx=2)

        ttk.Label(settings_tab, text="Глобальный провайдер:").pack(anchor="w", padx=6, pady=(8, 2))
        self.provider_var = tk.StringVar(value=self.state.selected_global_provider)
        self.provider_combo = ttk.Combobox(settings_tab, textvariable=self.provider_var, state="readonly")
        self.provider_combo.pack(fill="x", padx=6, pady=2)
        ttk.Label(settings_tab, text="Глобальная модель:").pack(anchor="w", padx=6, pady=(8, 2))
        self.model_var = tk.StringVar(value=self.state.selected_global_model)
        self.model_combo = ttk.Combobox(settings_tab, textvariable=self.model_var)
        self.model_combo.pack(fill="x", padx=6, pady=2)
        ttk.Button(settings_tab, text="Обновить список локальных моделей", command=self.refresh_model_status).pack(fill="x", padx=6, pady=4)
        ttk.Button(settings_tab, text="Проверить подключение", command=self.test_selected_provider).pack(fill="x", padx=6, pady=4)
        ttk.Button(settings_tab, text="Добавить/изменить API-ключ", command=self.add_provider).pack(fill="x", padx=6, pady=4)
        ttk.Button(settings_tab, text="Редактировать провайдера", command=self.edit_provider).pack(fill="x", padx=6, pady=4)
        ttk.Button(settings_tab, text="Удалить провайдера", command=self.delete_provider).pack(fill="x", padx=6, pady=4)
        ttk.Button(settings_tab, text="Включить/отключить провайдера", command=self.toggle_provider).pack(fill="x", padx=6, pady=4)
        ttk.Button(settings_tab, text="Открыть установку Ollama", command=self.open_ollama_download).pack(fill="x", padx=6, pady=4)
        ttk.Button(settings_tab, text="Запустить ollama serve", command=self.start_ollama_server).pack(fill="x", padx=6, pady=4)
        ttk.Button(settings_tab, text="Применить модель", command=self.apply_global_model).pack(fill="x", padx=6, pady=4)
        self.ollama_details = tk.Text(settings_tab, height=14, wrap="word", state="disabled")
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
            model = f" [{agent.provider_id}:{agent.model}]" if agent.model else ""
            fallback = f" ⇢ {agent.fallback_provider_id}:{agent.fallback_model}" if agent.fallback_provider_id and agent.fallback_model else ""
            self.agent_list.insert("end", f"{mark} {agent.name} — {agent.role}{model}{fallback}")
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

    def refresh_model_status(self) -> None:
        executable = self.ollama.executable_available()
        server = self.ollama.server_available()
        ollama_models = self.ollama.list_models() if server else []
        if self.state.providers and self.state.providers[0].id == "ollama":
            self.state.providers[0].models = ollama_models
        self.models = self.model_manager.all_model_labels()
        provider_ids = [provider.id for provider in self.state.providers]
        self.provider_combo.configure(values=provider_ids)
        self.model_combo.configure(values=self.models)
        if self.state.selected_global_provider not in provider_ids:
            self.state.selected_global_provider = "ollama"
        self.provider_var.set(self.state.selected_global_provider)
        if self.models and not self.model_var.get():
            first = ModelSelection.from_string(self.models[0])
            self.provider_var.set(first.provider_id)
            self.model_var.set(first.model)
        cloud_enabled = sum(1 for provider in self.state.providers if provider.id != "ollama" and provider.enabled)
        if server:
            self.status_var.set(f"Провайдеры: Ollama доступна ({len(ollama_models)} моделей), облачных включено: {cloud_enabled}")
        elif executable:
            self.status_var.set(f"Ollama установлена, но сервер не отвечает; облачных включено: {cloud_enabled}")
        else:
            self.status_var.set(f"Ollama не найдена; облачных включено: {cloud_enabled}")
        details = [
            "Локальные модели Ollama:",
            f" • Исполняемый файл: {'найден' if executable else 'не найден'}",
            f" • Сервер {OLLAMA_URL}: {'доступен' if server else 'недоступен'}",
            *(f" • {model}" for model in ollama_models),
            "",
            "Провайдеры моделей:",
        ]
        for provider in self.state.providers:
            status = self.model_manager.statuses.get(provider.id, ProviderStatus(provider_id=provider.id))
            key_text = "локальный" if provider.id == "ollama" else provider.masked_key()
            enabled = "включён" if provider.enabled else "отключён"
            models = provider.models or DEFAULT_PROVIDER_MODELS.get(provider.provider_type, [])
            details.append(
                f" • {provider.id} ({provider.name}, {provider.provider_type}) — {enabled}, ключ: {key_text}, "
                f"статус: {status.status}, модель: {status.active_model or provider.default_model or '-'}, "
                f"скорость: {status.last_latency_ms} мс, запросов: {status.request_count}"
            )
            if status.last_error:
                details.append(f"   ошибка: {status.last_error}")
            if models:
                details.append(f"   модели: {', '.join(models[:6])}{'…' if len(models) > 6 else ''}")
        self.ollama_details.configure(state="normal")
        self.ollama_details.delete("1.0", "end")
        self.ollama_details.insert("end", "\n".join(details))
        self.ollama_details.configure(state="disabled")
        self.refresh_agents()

    def _selected_provider(self) -> ProviderConfig | None:
        provider_id = self.provider_var.get() or self.state.selected_global_provider
        return self.model_manager.provider_config(provider_id)

    def add_provider(self) -> None:
        dialog = ProviderDialog(self, "Добавление провайдера", self.cipher)
        if dialog.result_provider:
            with self.lock:
                existing = self.model_manager.provider_config(dialog.result_provider.id)
                if existing:
                    self.state.providers = [dialog.result_provider if provider.id == existing.id else provider for provider in self.state.providers]
                else:
                    self.state.providers.append(dialog.result_provider)
                self.state.events.append(f"{now_iso()} — Добавлен/обновлён провайдер {dialog.result_provider.id}.")
            self.save_state()
            self.refresh_model_status()

    def edit_provider(self) -> None:
        provider = self._selected_provider()
        if not provider or provider.id == "ollama":
            messagebox.showinfo("Провайдер", "Выберите облачный провайдер для редактирования.", parent=self)
            return
        dialog = ProviderDialog(self, "Редактирование провайдера", self.cipher, provider)
        if dialog.result_provider:
            with self.lock:
                self.state.events.append(f"{now_iso()} — Изменён провайдер {dialog.result_provider.id}.")
            self.save_state()
            self.refresh_model_status()

    def delete_provider(self) -> None:
        provider = self._selected_provider()
        if not provider or provider.id == "ollama":
            messagebox.showinfo("Провайдер", "Встроенный Ollama удалить нельзя; выберите облачный провайдер.", parent=self)
            return
        if messagebox.askyesno("Удаление провайдера", f"Удалить провайдера {provider.name}?", parent=self):
            with self.lock:
                self.state.providers = [item for item in self.state.providers if item.id != provider.id]
                for agent in self.state.agents:
                    if agent.provider_id == provider.id:
                        agent.provider_id = "ollama"
                    if agent.fallback_provider_id == provider.id:
                        agent.fallback_provider_id = ""
                        agent.fallback_model = ""
                self.state.events.append(f"{now_iso()} — Удалён провайдер {provider.id}.")
            self.save_state()
            self.refresh_model_status()

    def toggle_provider(self) -> None:
        provider = self._selected_provider()
        if not provider:
            return
        if provider.id == "ollama":
            messagebox.showinfo("Провайдер", "Ollama управляется запуском локального сервера.", parent=self)
            return
        with self.lock:
            provider.enabled = not provider.enabled
            self.state.events.append(f"{now_iso()} — Провайдер {provider.id} {'включён' if provider.enabled else 'отключён'}.")
        self.save_state()
        self.refresh_model_status()

    def test_selected_provider(self) -> None:
        provider_id = self.provider_var.get() or "ollama"
        model = self.model_var.get().strip()
        ok, message, latency = self.model_manager.test_connection(provider_id, model)
        self.state.events.append(f"{now_iso()} — Проверка {provider_id}: {message} ({latency} мс).")
        self.save_state()
        self.refresh_model_status()
        title = "Подключение успешно" if ok else "Ошибка подключения"
        show = messagebox.showinfo if ok else messagebox.showerror
        show(title, f"{provider_id}: {message}\nСкорость ответа: {latency} мс", parent=self)

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
        dialog = AgentDialog(self, "Создание агента", models=self.models, providers=self.state.providers)
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
        dialog = AgentDialog(self, "Редактирование агента", agent=agent, models=self.models, providers=self.state.providers)
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
            self.state.selected_global_provider = self.provider_var.get() or "ollama"
            self.state.selected_global_model = self.model_var.get().strip()
            if ":" in self.state.selected_global_model:
                selection = ModelSelection.from_string(self.state.selected_global_model)
                self.state.selected_global_provider = selection.provider_id
                self.state.selected_global_model = selection.model
                self.provider_var.set(selection.provider_id)
                self.model_var.set(selection.model)
            self.state.events.append(f"{now_iso()} — Выбрана глобальная модель: {self.state.selected_global_provider}:{self.state.selected_global_model or 'не задана'}.")
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
            self.after(1500, self.refresh_model_status)
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
