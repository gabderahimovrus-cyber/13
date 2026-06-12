# Многоагентная чат-система с локальными и облачными моделями

Настольное Windows-приложение на русском языке для общения с несколькими автономными ИИ-агентами. Агенты работают через единый **Model Manager**, поэтому источник модели может быть локальным через Ollama или облачным через API-ключ.

## Возможности

- Русскоязычный GUI на Tkinter с главным окном чата.
- Единый менеджер моделей для всех агентов: агенты не обращаются напрямую к Ollama или облачным API.
- Одновременная поддержка локальных моделей Ollama и облачных провайдеров OpenAI, Anthropic, Google Gemini и OpenRouter.
- Раздел **«Модели»** для выбора глобального провайдера/модели, обновления списка локальных моделей, запуска Ollama и проверки подключений.
- Локальный режим Ollama: автоматическая проверка исполняемого файла, локального сервера `http://127.0.0.1:11434` и списка установленных моделей.
- Управление API-ключами облачных провайдеров: добавление, изменение, удаление, временное отключение и маскирование ключей после сохранения.
- Локальное шифрование API-ключей перед записью в `state.json`.
- Индивидуальная настройка каждого агента: имя, роль, цель, системный промпт, провайдер, модель, резервный провайдер/модель, лимит памяти и активность.
- Автоматическое переключение агента на резервную модель при ошибке выбранного провайдера.
- Отображение состояния провайдеров: доступность, текущая модель, последняя скорость ответа, число запросов и последняя ошибка.
- Краткосрочная и долгосрочная память каждого агента, общая история сообщений и симуляция работают одинаково независимо от провайдера модели.
- Устойчивый движок: ошибки модели и некорректные JSON-ответы логируются и не останавливают приложение.
- Возможность создания новых агентов действиями существующих агентов через структурированный ответ модели.

## Запуск

1. Установите Python 3.10+ для Windows.
2. Для локального режима установите Ollama: <https://ollama.com/download/windows>.
3. Загрузите локальную модель, например:

   ```powershell
   ollama pull llama3.1
   ```

4. Запустите приложение:

   ```powershell
   python main.py
   ```

5. Откройте вкладку **«Модели»**:
   - нажмите **«Обновить список локальных моделей»** для Ollama;
   - выберите глобальный провайдер и модель;
   - для облачных провайдеров добавьте API-ключ через **«Добавить/изменить API-ключ»**;
   - нажмите **«Проверить подключение»** перед использованием провайдера.

6. В разделе **«Агенты»** создайте или отредактируйте агента и назначьте ему собственные основной и резервный провайдеры/модели. Затем нажмите **«Возобновить»**, чтобы агенты начали циклическую обработку сообщений.

## Провайдеры моделей

Встроенные провайдеры:

- `ollama` — локальный сервер Ollama без API-ключа.
- `openai` — OpenAI Chat Completions API.
- `anthropic` — Anthropic Messages API.
- `gemini` — Google Gemini Generate Content API.
- `openrouter` — OpenRouter OpenAI-compatible API.

Архитектура провайдеров модульная: новые облачные интеграции добавляются реализацией провайдера в `ModelManager` без изменения логики агентов, памяти и симуляции.

## Хранение данных и ключей

Состояние сохраняется в `%APPDATA%\LocalMultiAgentChat\state.json`. Если файл повреждён, приложение создаёт резервную копию и запускается с начальным состоянием.

API-ключи сохраняются локально в зашифрованном виде. Секрет для локального шифрования находится в `%APPDATA%\LocalMultiAgentChat\.key_secret`. После сохранения ключи в интерфейсе показываются только в замаскированном виде.

## Формат действий агента

Каждый агент получает промпт с требованием вернуть JSON:

```json
{
  "respond": true,
  "message": "сообщение",
  "recipient": "Все",
  "memory": "важный вывод",
  "state_update": {"goal": "", "system_prompt": ""},
  "create_agent": null
}
```

Если модель вернула некорректный JSON, ответ сохраняется как обычное сообщение, а ошибка не прерывает работу системы.

## Self-modification API bridge

The project includes `self_mod_bridge.py`, a localhost-only bridge for controlled multi-agent code operations. It is intentionally disabled for write/execute operations until `SelfModificationBridge.confirm_guardrails()` returns a complete safety report.

Available high-level interfaces:

- **VFS Bridge**: `readFile(path)`, `writeFile(path, content, session)`, and `listDirectory(path)` operate only inside the project root.
- **Runtime API hooks**: `HotReloader.reload_module(module_name)` refreshes reloadable project modules without a full application restart.
- **Command executor**: `ShellWrapper.execute(args)` allows only dependency-management commands such as `pip`, `python -m pip`, and `npm`; pip commands are redirected into `.agent_bridge/venv`.
- **State persistence**: `AgentContextStore.save()` and `AgentContextStore.load()` preserve JSON agent identities and task state in `.agent_bridge/context.json`.
- **HTTP adapter**: `create_bridge_server(project_root)` exposes guarded JSON endpoints on localhost (`/guardrails/confirm`, `/vfs/read`, `/vfs/write`, `/command/execute`, `/rollback`) for an agent runtime that prefers API calls.
- **Health check monitor**: `HealthCheckMonitor` rejects deployment promotion when sampled latency exceeds the configured baseline or any sample returns a 5xx status.

Safety guardrails implemented before write/execute permissions are enabled:

1. **Kernel isolation** blocks bridge writes to protected core files and symbols, including `main.py` and chat/agent orchestration entry points.
2. **Recursive loop guard** limits writes per `BridgeSession`; exceeding the limit restores snapshots from that session.
3. **Pre-commit linting** parses Python and JSON content before it is written, rejecting syntax errors before application.
4. **Emergency rollback** creates mandatory snapshots before every write and restores the latest snapshot with the one-word command `ROLLBACK`.
5. **Principle of least privilege** resolves all file paths against the project root and rejects traversal outside it.
6. **Isolated execution sandbox** runs allow-listed commands with timeout plus CPU/RAM limits where the platform supports POSIX resource controls.
7. **Atomic deployment** stages blue/green deployment slots under `.agent_bridge/deployments` and switches the active slot only after a health check passes.
8. **Dependency isolation** keeps Python dependency changes inside `.agent_bridge/venv` instead of the core interpreter environment.
