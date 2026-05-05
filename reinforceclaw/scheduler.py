"""Training scheduler. Uses launchd on mac, systemd on linux."""
# default: "auto" queues background training when a batch is ready.
# scheduled times like "03:00" are optional for users who prefer a clock window.

import os
import platform
import secrets
import subprocess
import sys
from datetime import datetime, timedelta
from html import escape as xml_escape
from pathlib import Path

from . import db

PLIST_PATH = Path.home() / "Library/LaunchAgents/com.reinforceclaw.train.plist"
SYSTEMD_PATH = Path.home() / ".config/systemd/user/reinforceclaw-train.timer"
SYSTEMD_SERVICE = Path.home() / ".config/systemd/user/reinforceclaw-train.service"
BRIDGE_PLIST_PATH = Path.home() / "Library/LaunchAgents/com.reinforceclaw.openclaw-bridge.plist"
BRIDGE_SYSTEMD_SERVICE = Path.home() / ".config/systemd/user/reinforceclaw-openclaw-bridge.service"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_LOG = Path.home() / ".reinforceclaw" / "train.log"
BRIDGE_LOG = Path.home() / ".reinforceclaw" / "openclaw-bridge.log"
DEFAULT_WINDOW_MINUTES = 180
LAST_ERROR = ""
_HF_ENV_KEYS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN")
_RUNTIME_ENV_KEYS = (
    "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
    "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy",
    "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "REINFORCECLAW_ALLOW_REMOTE_CODE",
)


def _attempt_times(schedule="03:00", window_minutes: int = DEFAULT_WINDOW_MINUTES):
    # Windows can wrap past midnight; launchd/systemd then run those tries on the next day.
    hour, minute = _parse_time(schedule)
    start = datetime(2000, 1, 1, hour, minute)
    tries = max(1, int((window_minutes - 1) // 60) + 1)
    return [((start + timedelta(hours=i)).hour, (start + timedelta(hours=i)).minute) for i in range(tries)]


def install(schedule="03:00", window_minutes: int = DEFAULT_WINDOW_MINUTES):
    """Install system scheduler. Returns True on success."""
    db.secure_private_dir(TRAIN_LOG.parent)
    if schedule == "manual":
        return uninstall()
    if schedule == "auto":
        return uninstall()  # no system scheduler needed, hooks handle it

    attempt_times = _attempt_times(schedule, window_minutes)
    if platform.system() == "Darwin":
        return _install_launchd(attempt_times)
    else:
        return _install_systemd(attempt_times)


def uninstall():
    """Remove system scheduler."""
    ok = True
    system = platform.system()
    if PLIST_PATH.exists():
        if system == "Darwin":
            ok = _run_ok(["launchctl", "unload", str(PLIST_PATH)]) and ok
        PLIST_PATH.unlink(missing_ok=True)
    if SYSTEMD_PATH.exists() or SYSTEMD_SERVICE.exists():
        if system != "Darwin":
            ok = _run_ok(["systemctl", "--user", "disable", "--now", "reinforceclaw-train.timer"]) and ok
            ok = _run_ok(["systemctl", "--user", "stop", "reinforceclaw-train.service"]) and ok
        SYSTEMD_PATH.unlink(missing_ok=True)
        SYSTEMD_SERVICE.unlink(missing_ok=True)
        if system != "Darwin":
            ok = _run_ok(["systemctl", "--user", "daemon-reload"]) and ok
    return ok


def install_openclaw_bridge():
    """Install and start the local OpenClaw bridge service."""
    db.secure_private_dir(BRIDGE_LOG.parent)
    if platform.system() == "Darwin":
        return _install_openclaw_bridge_launchd()
    if platform.system() == "Windows":
        global LAST_ERROR
        LAST_ERROR = "automatic OpenClaw bridge service is not supported on Windows yet"
        return False
    return _install_openclaw_bridge_systemd()


def uninstall_openclaw_bridge():
    """Remove the local OpenClaw bridge service."""
    ok = True
    system = platform.system()
    if BRIDGE_PLIST_PATH.exists():
        if system == "Darwin":
            ok = _run_ok(["launchctl", "unload", str(BRIDGE_PLIST_PATH)]) and ok
        BRIDGE_PLIST_PATH.unlink(missing_ok=True)
    if BRIDGE_SYSTEMD_SERVICE.exists():
        if system != "Darwin":
            ok = _run_ok(["systemctl", "--user", "disable", "--now", "reinforceclaw-openclaw-bridge.service"]) and ok
            ok = _run_ok(["systemctl", "--user", "daemon-reload"]) and ok
        BRIDGE_SYSTEMD_SERVICE.unlink(missing_ok=True)
    return ok


def _run_ok(cmd):
    global LAST_ERROR
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return True
        LAST_ERROR = (result.stderr or result.stdout or f"{cmd[0]} exited {result.returncode}").strip()
        return False
    except subprocess.TimeoutExpired:
        LAST_ERROR = f"{cmd[0]} timed out"
        return False
    except OSError as exc:
        LAST_ERROR = str(exc)
        return False


def _write_text(path, text):
    path = Path(path).expanduser()
    target = path.resolve(strict=False) if path.is_symlink() else path
    if target == path:
        db.secure_private_dir(path.parent)
    tmp = target.with_name(f".{target.name}.{os.getpid()}-{secrets.token_hex(16)}.tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        tmp.replace(target)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _snapshot(*paths):
    return {
        path: (path.resolve(strict=False) if path.is_symlink() else path).read_text(encoding="utf-8") if path.exists() else None
        for path in paths
    }


def _restore(snapshot):
    for path, text in snapshot.items():
        if text is None:
            path.unlink(missing_ok=True)
        else:
            _write_text(path, text)


def _systemd_env(value):
    return str(value).replace("\r", " ").replace("\n", " ").replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%").replace("$", "$$")


def _systemd_arg(value):
    return f'"{_systemd_env(value)}"'


def _scheduler_env():
    env = {
        key: os.environ[key]
        for key in _RUNTIME_ENV_KEYS
        if os.environ.get(key) and not (key.lower().endswith("_proxy") and "@" in os.environ[key])
    }
    token_path = Path.home() / ".cache/huggingface/token"
    if token_path.exists() or os.environ.get("REINFORCECLAW_SCHEDULER_PERSIST_HF_TOKEN") != "1":
        return env
    env.update({key: os.environ[key] for key in _HF_ENV_KEYS if os.environ.get(key)})
    return env


def _parse_time(s):
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError("time must be HH:MM")
    try:
        hour, minute = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError("time must be HH:MM") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError("time must be HH:MM")
    return hour, minute


def _install_launchd(attempt_times):
    env_xml = "\n".join(
        f"        <key>{xml_escape(key)}</key>\n        <string>{xml_escape(value)}</string>"
        for key, value in {"PYTHONPATH": str(PROJECT_ROOT), **_scheduler_env()}.items()
    )
    intervals = "\n".join(
        f"""        <dict>
            <key>Hour</key>
            <integer>{hour}</integer>
            <key>Minute</key>
            <integer>{minute}</integer>
        </dict>"""
        for hour, minute in attempt_times
    )
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.reinforceclaw.train</string>
    <key>ProgramArguments</key>
    <array>
        <string>{xml_escape(sys.executable)}</string>
        <string>-m</string>
        <string>reinforceclaw.cli</string>
        <string>train</string>
        <string>--background</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{xml_escape(str(PROJECT_ROOT))}</string>
    <key>EnvironmentVariables</key>
    <dict>
{env_xml}
    </dict>
    <key>StartCalendarInterval</key>
    <array>
{intervals}
    </array>
    <key>StandardOutPath</key>
    <string>{xml_escape(str(TRAIN_LOG))}</string>
    <key>StandardErrorPath</key>
    <string>{xml_escape(str(TRAIN_LOG))}</string>
</dict>
</plist>"""
    db.secure_private_dir(PLIST_PATH.parent)
    before = _snapshot(PLIST_PATH)
    if PLIST_PATH.exists() and not _run_ok(["launchctl", "unload", str(PLIST_PATH)]):
        return False
    _write_text(PLIST_PATH, plist)
    if _run_ok(["launchctl", "load", str(PLIST_PATH)]):
        return True
    _restore(before)
    if before[PLIST_PATH] is not None:
        _run_ok(["launchctl", "load", str(PLIST_PATH)])
    return False


def _install_systemd(attempt_times):
    db.secure_private_dir(SYSTEMD_PATH.parent)
    before = _snapshot(SYSTEMD_SERVICE, SYSTEMD_PATH)
    if SYSTEMD_PATH.exists() and not _run_ok(["systemctl", "--user", "disable", "--now", "reinforceclaw-train.timer"]):
        return False
    # service
    _write_text(SYSTEMD_SERVICE, f"""[Unit]
Description=ReinforceClaw RL training

[Service]
ExecStart={_systemd_arg(sys.executable)} -m reinforceclaw.cli train --background
WorkingDirectory={_systemd_arg(PROJECT_ROOT)}
Environment="PYTHONPATH={_systemd_env(PROJECT_ROOT)}"
{chr(10).join(f'Environment="{_systemd_env(k)}={_systemd_env(v)}"' for k, v in _scheduler_env().items())}
StandardOutput=append:{_systemd_arg(TRAIN_LOG)}
StandardError=append:{_systemd_arg(TRAIN_LOG)}
""")
    calendar = "\n".join(
        f"OnCalendar=*-*-* {hour:02d}:{minute:02d}:00"
        for hour, minute in attempt_times
    )
    _write_text(SYSTEMD_PATH, f"""[Unit]
Description=ReinforceClaw daily training

[Timer]
{calendar}
Persistent=true

[Install]
WantedBy=timers.target
""")
    ok = _run_ok(["systemctl", "--user", "daemon-reload"]) and _run_ok(["systemctl", "--user", "enable", "--now", "reinforceclaw-train.timer"])
    if ok:
        return True
    _restore(before)
    _run_ok(["systemctl", "--user", "daemon-reload"])
    if before[SYSTEMD_PATH] is not None:
        _run_ok(["systemctl", "--user", "enable", "--now", "reinforceclaw-train.timer"])
    return False


def _service_env_xml():
    return "\n".join(
        f"        <key>{xml_escape(key)}</key>\n        <string>{xml_escape(value)}</string>"
        for key, value in {"PYTHONPATH": str(PROJECT_ROOT), **_scheduler_env()}.items()
    )


def _service_env_systemd():
    return "\n".join(f'Environment="{_systemd_env(k)}={_systemd_env(v)}"' for k, v in _scheduler_env().items())


def _install_openclaw_bridge_launchd():
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.reinforceclaw.openclaw-bridge</string>
    <key>ProgramArguments</key>
    <array>
        <string>{xml_escape(sys.executable)}</string>
        <string>-m</string>
        <string>reinforceclaw.hooks.openclaw</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{xml_escape(str(PROJECT_ROOT))}</string>
    <key>EnvironmentVariables</key>
    <dict>
{_service_env_xml()}
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{xml_escape(str(BRIDGE_LOG))}</string>
    <key>StandardErrorPath</key>
    <string>{xml_escape(str(BRIDGE_LOG))}</string>
</dict>
</plist>"""
    db.secure_private_dir(BRIDGE_PLIST_PATH.parent)
    before = _snapshot(BRIDGE_PLIST_PATH)
    if BRIDGE_PLIST_PATH.exists() and not _run_ok(["launchctl", "unload", str(BRIDGE_PLIST_PATH)]):
        return False
    _write_text(BRIDGE_PLIST_PATH, plist)
    if _run_ok(["launchctl", "load", str(BRIDGE_PLIST_PATH)]):
        return True
    _restore(before)
    if before[BRIDGE_PLIST_PATH] is not None:
        _run_ok(["launchctl", "load", str(BRIDGE_PLIST_PATH)])
    return False


def _install_openclaw_bridge_systemd():
    db.secure_private_dir(BRIDGE_SYSTEMD_SERVICE.parent)
    before = _snapshot(BRIDGE_SYSTEMD_SERVICE)
    if BRIDGE_SYSTEMD_SERVICE.exists() and not _run_ok(["systemctl", "--user", "disable", "--now", "reinforceclaw-openclaw-bridge.service"]):
        return False
    _write_text(BRIDGE_SYSTEMD_SERVICE, f"""[Unit]
Description=ReinforceClaw OpenClaw bridge

[Service]
ExecStart={_systemd_arg(sys.executable)} -m reinforceclaw.hooks.openclaw
WorkingDirectory={_systemd_arg(PROJECT_ROOT)}
Environment="PYTHONPATH={_systemd_env(PROJECT_ROOT)}"
{_service_env_systemd()}
Restart=always
RestartSec=3
StandardOutput=append:{_systemd_arg(BRIDGE_LOG)}
StandardError=append:{_systemd_arg(BRIDGE_LOG)}

[Install]
WantedBy=default.target
""")
    ok = _run_ok(["systemctl", "--user", "daemon-reload"]) and _run_ok(["systemctl", "--user", "enable", "--now", "reinforceclaw-openclaw-bridge.service"])
    if ok:
        return True
    _restore(before)
    _run_ok(["systemctl", "--user", "daemon-reload"])
    if before[BRIDGE_SYSTEMD_SERVICE] is not None:
        _run_ok(["systemctl", "--user", "enable", "--now", "reinforceclaw-openclaw-bridge.service"])
    return False
