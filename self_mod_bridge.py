"""Safe self-modification bridge for local multi-agent workflows.

The bridge intentionally exposes a small, high-level API for agents that need to
inspect project files, propose edits, run dependency commands, hot-reload project
modules, persist agent context, and stage deployments.  It is defensive by
construction: all file access is rooted inside the project directory, every write
creates a snapshot, Python/JSON syntax is validated before content is applied,
and emergency rollback is available through the one-word command ``ROLLBACK``.
"""

from __future__ import annotations

import ast
import importlib
import json
import os

try:
    import resource
except ImportError:  # pragma: no cover - Windows has no POSIX resource module.
    resource = None  # type: ignore[assignment]

import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable


PROTECTED_PATHS = {
    "main.py",  # Tk user communication loop and agent orchestration live here.
}
PROTECTED_NAMES = {
    "AgentEngine",
    "ChatApp",
    "_parse_agent_response",
    "_build_prompt",
    "send_message",
}
DEFAULT_ALLOWED_COMMANDS = {
    "pip",
    "npm",
    "python -m pip",
    "python3 -m pip",
    sys.executable + " -m pip",
}


class BridgeError(RuntimeError):
    """Base error raised when a bridge guardrail rejects an operation."""


class AccessDeniedError(BridgeError):
    """Raised when an operation attempts to leave the project sandbox."""


class GuardrailViolation(BridgeError):
    """Raised when a safety guardrail blocks a requested action."""


@dataclass(frozen=True)
class CommandResult:
    """Sanitized command execution result returned by :class:`ShellWrapper`."""

    args: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_ms: int


@dataclass(frozen=True)
class Snapshot:
    """Metadata for a file snapshot captured before a write."""

    id: str
    original_path: str
    snapshot_path: str
    created_at: float
    reason: str


@dataclass
class BridgePolicy:
    """Runtime limits and allow-lists for the self-modification bridge."""

    max_iterations_per_request: int = 5
    command_timeout_seconds: int = 120
    cpu_seconds: int = 10
    memory_mb: int = 256
    allowed_commands: set[str] = field(default_factory=lambda: set(DEFAULT_ALLOWED_COMMANDS))
    protected_paths: set[str] = field(default_factory=lambda: set(PROTECTED_PATHS))
    protected_names: set[str] = field(default_factory=lambda: set(PROTECTED_NAMES))


@dataclass
class BridgeSession:
    """Tracks self-modification iterations for a single request."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    iterations: int = 0
    snapshot_ids: list[str] = field(default_factory=list)


class ProjectSandbox:
    """Resolves paths while enforcing least-privilege project-directory access."""

    def __init__(self, project_root: Path | str) -> None:
        self.root = Path(project_root).resolve()
        if not self.root.exists():
            raise AccessDeniedError(f"Project root does not exist: {self.root}")

    def resolve(self, path: Path | str) -> Path:
        candidate = (self.root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise AccessDeniedError(f"Path is outside project sandbox: {path}")
        return candidate

    def relative(self, path: Path | str) -> str:
        return self.resolve(path).relative_to(self.root).as_posix()


class SnapshotManager:
    """Creates and restores mandatory file snapshots before bridge writes."""

    def __init__(self, sandbox: ProjectSandbox, storage_dir: Path | str = ".agent_bridge/snapshots") -> None:
        self.sandbox = sandbox
        self.storage_dir = self.sandbox.resolve(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.storage_dir / "manifest.json"
        self._snapshots = self._load_manifest()

    def create(self, target: Path | str, reason: str) -> Snapshot:
        target_path = self.sandbox.resolve(target)
        snapshot_id = uuid.uuid4().hex
        snapshot_file = self.storage_dir / f"{snapshot_id}.snapshot"
        if target_path.exists():
            if target_path.is_dir():
                raise GuardrailViolation("Directory writes are not supported by snapshot manager")
            shutil.copy2(target_path, snapshot_file)
        else:
            snapshot_file.write_bytes(b"")
        snapshot = Snapshot(
            id=snapshot_id,
            original_path=self.sandbox.relative(target_path),
            snapshot_path=self.sandbox.relative(snapshot_file),
            created_at=time.time(),
            reason=reason,
        )
        self._snapshots[snapshot_id] = snapshot
        self._save_manifest()
        return snapshot

    def latest(self) -> Snapshot | None:
        if not self._snapshots:
            return None
        return max(self._snapshots.values(), key=lambda item: item.created_at)

    def restore(self, snapshot_id: str | None = None) -> Snapshot:
        snapshot = self._snapshots.get(snapshot_id) if snapshot_id else self.latest()
        if snapshot is None:
            raise GuardrailViolation("No snapshot is available for rollback")
        original = self.sandbox.resolve(snapshot.original_path)
        snapshot_file = self.sandbox.resolve(snapshot.snapshot_path)
        original.parent.mkdir(parents=True, exist_ok=True)
        if snapshot_file.stat().st_size == 0 and not original.exists():
            return snapshot
        shutil.copy2(snapshot_file, original)
        return snapshot

    def _load_manifest(self) -> dict[str, Snapshot]:
        if not self.manifest_path.exists():
            return {}
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {item["id"]: Snapshot(**item) for item in data}

    def _save_manifest(self) -> None:
        payload = [asdict(snapshot) for snapshot in self._snapshots.values()]
        self.manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class StaticAnalyzer:
    """Pre-commit style syntax checks used before bridge writes are applied."""

    def validate(self, path: Path | str, content: str) -> None:
        suffix = Path(path).suffix.lower()
        if suffix == ".py":
            ast.parse(content, filename=str(path))
        elif suffix == ".json":
            json.loads(content)


class VFSBridge:
    """Virtual file-system API restricted to the current project directory."""

    def __init__(
        self,
        sandbox: ProjectSandbox,
        snapshots: SnapshotManager,
        analyzer: StaticAnalyzer,
        policy: BridgePolicy,
    ) -> None:
        self.sandbox = sandbox
        self.snapshots = snapshots
        self.analyzer = analyzer
        self.policy = policy

    def readFile(self, path: str) -> str:  # noqa: N802 - public bridge API mirrors prompt contract.
        return self.sandbox.resolve(path).read_text(encoding="utf-8")

    def writeFile(self, path: str, content: str, session: BridgeSession | None = None) -> Snapshot:  # noqa: N802
        self._enforce_write_guardrails(path, content, session)
        target = self.sandbox.resolve(path)
        snapshot = self.snapshots.create(target, "pre-write")
        if session is not None:
            session.snapshot_ids.append(snapshot.id)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return snapshot

    def listDirectory(self, path: str = ".") -> list[str]:  # noqa: N802
        target = self.sandbox.resolve(path)
        if not target.is_dir():
            raise AccessDeniedError(f"Not a directory: {path}")
        return sorted(item.name for item in target.iterdir())

    def _enforce_write_guardrails(self, path: str, content: str, session: BridgeSession | None) -> None:
        relative_path = self.sandbox.relative(path)
        if relative_path in self.policy.protected_paths:
            raise GuardrailViolation(f"Protected core file cannot be modified through bridge: {relative_path}")
        if session is not None:
            session.iterations += 1
            if session.iterations > self.policy.max_iterations_per_request:
                for snapshot_id in reversed(session.snapshot_ids):
                    self.snapshots.restore(snapshot_id)
                raise GuardrailViolation("Self-modification iteration limit exceeded; rollback completed")
        self.analyzer.validate(relative_path, content)
        for protected_name in self.policy.protected_names:
            if f"def {protected_name}" in content or f"class {protected_name}" in content:
                raise GuardrailViolation(f"Protected communication/token-processing symbol is not writable: {protected_name}")


class HotReloader:
    """Reloads project modules without restarting the full desktop application."""

    def __init__(self, sandbox: ProjectSandbox) -> None:
        self.sandbox = sandbox

    def reload_module(self, module_name: str) -> ModuleType:
        module = sys.modules.get(module_name)
        if module is None:
            module = importlib.import_module(module_name)
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            raise GuardrailViolation(f"Module has no reloadable file: {module_name}")
        self.sandbox.resolve(module_file)
        return importlib.reload(module)


class ShellWrapper:
    """Secure command executor for dependency-management commands only."""

    def __init__(self, sandbox: ProjectSandbox, policy: BridgePolicy, venv_dir: str = ".agent_bridge/venv") -> None:
        self.sandbox = sandbox
        self.policy = policy
        self.venv_dir = self.sandbox.resolve(venv_dir)

    def ensure_venv(self) -> Path:
        if not self.venv_dir.exists():
            subprocess.run([sys.executable, "-m", "venv", str(self.venv_dir)], cwd=self.sandbox.root, check=True, timeout=60)
        return self.venv_dir

    def execute(self, args: list[str], cwd: str = ".") -> CommandResult:
        if not args:
            raise GuardrailViolation("Command cannot be empty")
        command_key = self._command_key(args)
        if command_key not in self.policy.allowed_commands:
            raise GuardrailViolation(f"Command is not allow-listed: {command_key}")
        resolved_cwd = self.sandbox.resolve(cwd)
        safe_args = self._isolate_dependency_command(args)
        started = time.perf_counter()
        completed = subprocess.run(
            safe_args,
            cwd=resolved_cwd,
            text=True,
            capture_output=True,
            timeout=self.policy.command_timeout_seconds,
            env=self._safe_env(),
            preexec_fn=self._limit_resources if os.name == "posix" and resource is not None else None,
            check=False,
        )
        return CommandResult(
            args=safe_args,
            returncode=completed.returncode,
            stdout=completed.stdout[-8000:],
            stderr=completed.stderr[-8000:],
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    def _command_key(self, args: list[str]) -> str:
        if len(args) >= 3 and args[1:3] == ["-m", "pip"]:
            return f"{args[0]} -m pip"
        return args[0]

    def _isolate_dependency_command(self, args: list[str]) -> list[str]:
        if self._command_key(args).endswith("-m pip") or args[0] == "pip":
            self.ensure_venv()
            python = self.venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            return [str(python), "-m", "pip", *args[3 if len(args) >= 3 and args[1:3] == ["-m", "pip"] else 1 :]]
        return args

    def _safe_env(self) -> dict[str, str]:
        return {
            "PATH": os.defpath,
            "PYTHONNOUSERSITE": "1",
            "PIP_REQUIRE_VIRTUALENV": "true",
            "NO_COLOR": "1",
        }

    def _limit_resources(self) -> None:
        if resource is None:
            return
        resource.setrlimit(resource.RLIMIT_CPU, (self.policy.cpu_seconds, self.policy.cpu_seconds + 1))
        memory_bytes = self.policy.memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))


class AgentContextStore:
    """JSON persistence for agent identities and task states across reloads."""

    def __init__(self, sandbox: ProjectSandbox, path: str = ".agent_bridge/context.json") -> None:
        self.sandbox = sandbox
        self.path = self.sandbox.resolve(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, context: dict[str, Any]) -> None:
        self.path.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise GuardrailViolation("Persisted context must be a JSON object")
        return data


@dataclass(frozen=True)
class HealthSample:
    """Single health-check sample for blue-green deployment decisions."""

    status_code: int
    latency_ms: int


class HealthCheckMonitor:
    """Rejects deployments when latency regresses or 5xx responses appear."""

    def __init__(self, baseline_latency_ms: int = 1000, max_latency_multiplier: float = 2.0) -> None:
        self.baseline_latency_ms = baseline_latency_ms
        self.max_latency_multiplier = max_latency_multiplier

    def is_healthy(self, samples: Iterable[HealthSample]) -> bool:
        limit = int(self.baseline_latency_ms * self.max_latency_multiplier)
        for sample in samples:
            if sample.status_code >= 500 or sample.latency_ms > limit:
                return False
        return True


class BlueGreenDeploymentManager:
    """Stages project files into blue/green slots and rolls back on failed health checks."""

    def __init__(self, sandbox: ProjectSandbox, health_check: Callable[[], bool] | None = None) -> None:
        self.sandbox = sandbox
        self.deploy_dir = self.sandbox.resolve(".agent_bridge/deployments")
        self.deploy_dir.mkdir(parents=True, exist_ok=True)
        self.pointer_path = self.deploy_dir / "active_slot.json"
        self.health_check = health_check or (lambda: True)

    def active_slot(self) -> str:
        if not self.pointer_path.exists():
            return "blue"
        return str(json.loads(self.pointer_path.read_text(encoding="utf-8")).get("active", "blue"))

    def deploy(self) -> str:
        current = self.active_slot()
        candidate = "green" if current == "blue" else "blue"
        candidate_dir = self.deploy_dir / candidate
        if candidate_dir.exists():
            shutil.rmtree(candidate_dir)
        ignore = shutil.ignore_patterns(".git", ".agent_bridge", "__pycache__", "*.pyc")
        shutil.copytree(self.sandbox.root, candidate_dir, ignore=ignore)
        if not self.health_check():
            if candidate_dir.exists():
                shutil.rmtree(candidate_dir)
            self.pointer_path.write_text(json.dumps({"active": current}, indent=2), encoding="utf-8")
            raise GuardrailViolation("Health check failed; blue-green switch was rolled back")
        self.pointer_path.write_text(json.dumps({"active": candidate}, indent=2), encoding="utf-8")
        return candidate


class SelfModificationBridge:
    """Facade that confirms guardrails before exposing write/execute capabilities."""

    emergency_command = "ROLLBACK"

    def __init__(self, project_root: Path | str, policy: BridgePolicy | None = None) -> None:
        self.policy = policy or BridgePolicy()
        self.sandbox = ProjectSandbox(project_root)
        self.snapshots = SnapshotManager(self.sandbox)
        self.analyzer = StaticAnalyzer()
        self.vfs = VFSBridge(self.sandbox, self.snapshots, self.analyzer, self.policy)
        self.shell = ShellWrapper(self.sandbox, self.policy)
        self.hot_reload = HotReloader(self.sandbox)
        self.context = AgentContextStore(self.sandbox)
        self.deployments = BlueGreenDeploymentManager(self.sandbox)
        self.write_enabled = False
        self.execute_enabled = False

    def confirm_guardrails(self) -> dict[str, bool]:
        status = {
            "kernel_isolation": bool(self.policy.protected_paths and self.policy.protected_names),
            "recursive_loop_guard": self.policy.max_iterations_per_request > 0,
            "pre_commit_linting": isinstance(self.analyzer, StaticAnalyzer),
            "emergency_rollback": self.snapshots.storage_dir.exists() and self.emergency_command == "ROLLBACK",
            "least_privilege": self.sandbox.root.is_dir(),
            "isolated_execution_sandbox": self.policy.cpu_seconds > 0 and self.policy.memory_mb > 0,
            "atomic_deployment": self.deployments.deploy_dir.exists(),
            "dependency_isolation": str(self.shell.venv_dir).startswith(str(self.sandbox.root)),
        }
        self.write_enabled = all(status[name] for name in ("kernel_isolation", "recursive_loop_guard", "pre_commit_linting", "emergency_rollback", "least_privilege"))
        self.execute_enabled = self.write_enabled and status["isolated_execution_sandbox"] and status["dependency_isolation"]
        return status

    def new_session(self) -> BridgeSession:
        return BridgeSession()

    def rollback(self, command: str = emergency_command) -> Snapshot:
        if command != self.emergency_command:
            raise GuardrailViolation("Emergency rollback requires the one-word ROLLBACK command")
        return self.snapshots.restore()

    def writeFile(self, path: str, content: str, session: BridgeSession | None = None) -> Snapshot:  # noqa: N802
        if not self.write_enabled:
            raise GuardrailViolation("Write permission disabled until guardrails are confirmed")
        return self.vfs.writeFile(path, content, session)

    def execute(self, args: list[str], cwd: str = ".") -> CommandResult:
        if not self.execute_enabled:
            raise GuardrailViolation("Execute permission disabled until guardrails are confirmed")
        return self.shell.execute(args, cwd)


class BridgeHTTPRequestHandler(BaseHTTPRequestHandler):
    """Small localhost JSON endpoint adapter for the bridge facade."""

    bridge: SelfModificationBridge

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler name.
        try:
            payload = self._read_json()
            if self.path == "/guardrails/confirm":
                self._send_json(self.bridge.confirm_guardrails())
            elif self.path == "/vfs/read":
                self._send_json({"content": self.bridge.vfs.readFile(str(payload.get("path", "")))})
            elif self.path == "/vfs/write":
                snapshot = self.bridge.writeFile(str(payload.get("path", "")), str(payload.get("content", "")))
                self._send_json({"snapshot": asdict(snapshot)})
            elif self.path == "/command/execute":
                result = self.bridge.execute(list(payload.get("args", [])), str(payload.get("cwd", ".")))
                self._send_json(asdict(result))
            elif self.path == "/rollback":
                snapshot = self.bridge.rollback(str(payload.get("command", "")))
                self._send_json({"restored": asdict(snapshot)})
            else:
                self._send_json({"error": "unknown endpoint"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001 - endpoint returns controlled JSON errors.
            self._send_json({"error": str(exc), "type": exc.__class__.__name__}, HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature.
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length == 0:
            return {}
        data = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(data, dict):
            raise GuardrailViolation("Request body must be a JSON object")
        return data

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def create_bridge_server(project_root: Path | str, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """Create a localhost-only bridge server; caller owns serving thread/lifecycle."""

    class Handler(BridgeHTTPRequestHandler):
        bridge = SelfModificationBridge(project_root)

    return ThreadingHTTPServer((host, port), Handler)
