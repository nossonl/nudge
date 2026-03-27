"""Training scheduler. Uses launchd on mac, systemd on linux."""
# default: train at 3am if batch is ready
# if machine was off at 3am, trains on next wake/boot
# configurable: "03:00", "manual", "auto" (immediate when batch ready)

import platform
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

PLIST_PATH = Path.home() / "Library/LaunchAgents/com.reinforceclaw.train.plist"
SYSTEMD_PATH = Path.home() / ".config/systemd/user/reinforceclaw-train.timer"
SYSTEMD_SERVICE = Path.home() / ".config/systemd/user/reinforceclaw-train.service"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_LOG = Path.home() / ".reinforceclaw" / "train.log"
DEFAULT_WINDOW_MINUTES = 180


def _attempt_times(schedule="03:00", window_minutes: int = DEFAULT_WINDOW_MINUTES):
    hour, minute = _parse_time(schedule)
    start = datetime(2000, 1, 1, hour, minute)
    tries = max(1, int((window_minutes - 1) // 60) + 1)
    return [((start + timedelta(hours=i)).hour, (start + timedelta(hours=i)).minute) for i in range(tries)]


def install(schedule="03:00", window_minutes: int = DEFAULT_WINDOW_MINUTES):
    """Install system scheduler. Returns True on success."""
    if schedule == "manual":
        uninstall()
        return True
    if schedule == "auto":
        uninstall()  # no system scheduler needed, hooks handle it
        return True

    attempt_times = _attempt_times(schedule, window_minutes)
    if platform.system() == "Darwin":
        return _install_launchd(attempt_times)
    else:
        return _install_systemd(attempt_times)


def uninstall():
    """Remove system scheduler."""
    if PLIST_PATH.exists():
        subprocess.run(["launchctl", "unload", str(PLIST_PATH)], capture_output=True)
        PLIST_PATH.unlink()
    if SYSTEMD_PATH.exists():
        subprocess.run(["systemctl", "--user", "disable", "--now", "reinforceclaw-train.timer"], capture_output=True)
        subprocess.run(["systemctl", "--user", "stop", "reinforceclaw-train.service"], capture_output=True)
        SYSTEMD_PATH.unlink()
        SYSTEMD_SERVICE.unlink(missing_ok=True)
        subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)


def _parse_time(s):
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError("time must be HH:MM")
    hour, minute = int(parts[0]), int(parts[1])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError("time must be HH:MM")
    return hour, minute


def _install_launchd(attempt_times):
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
        <string>{sys.executable}</string>
        <string>-m</string>
        <string>reinforceclaw.cli</string>
        <string>train</string>
        <string>--background</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{PROJECT_ROOT}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONPATH</key>
        <string>{PROJECT_ROOT}</string>
    </dict>
    <key>StartCalendarInterval</key>
    <array>
{intervals}
    </array>
    <key>StandardOutPath</key>
    <string>{TRAIN_LOG}</string>
    <key>StandardErrorPath</key>
    <string>{TRAIN_LOG}</string>
</dict>
</plist>"""
    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["launchctl", "unload", str(PLIST_PATH)], capture_output=True)
    PLIST_PATH.write_text(plist)
    r = subprocess.run(["launchctl", "load", str(PLIST_PATH)], capture_output=True)
    return r.returncode == 0


def _install_systemd(attempt_times):
    SYSTEMD_PATH.parent.mkdir(parents=True, exist_ok=True)
    # service
    SYSTEMD_SERVICE.write_text(f"""[Unit]
Description=ReinforceClaw RL training

[Service]
ExecStart={sys.executable} -m reinforceclaw.cli train --background
WorkingDirectory={PROJECT_ROOT}
Environment=PYTHONPATH={PROJECT_ROOT}
StandardOutput=append:{TRAIN_LOG}
StandardError=append:{TRAIN_LOG}
""")
    calendar = "\n".join(
        f"OnCalendar=*-*-* {hour:02d}:{minute:02d}:00"
        for hour, minute in attempt_times
    )
    SYSTEMD_PATH.write_text(f"""[Unit]
Description=ReinforceClaw daily training

[Timer]
{calendar}
Persistent=false

[Install]
WantedBy=timers.target
""")
    subprocess.run(["systemctl", "--user", "disable", "--now", "reinforceclaw-train.timer"], capture_output=True)
    subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
    r = subprocess.run(["systemctl", "--user", "enable", "--now", "reinforceclaw-train.timer"], capture_output=True)
    return r.returncode == 0
