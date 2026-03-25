"""Terminal feedback panel. Opens /dev/tty directly so it works inside hook subprocesses."""
# hooks get stdin piped from the parent process (JSON), so sys.stdin is NOT a terminal.
# we bypass that entirely by opening /dev/tty — the real terminal — for keypress reads.

import os
import sys
import select
from typing import Optional, Union

_KEYS = {"1": 1, "2": -1, "3": None}  # good, bad, skip


def _open_tty():
    try:
        return os.open("/dev/tty", os.O_RDONLY)
    except OSError:
        return None


def _raw_read(fd: int, timeout=0.1) -> Optional[str]:
    """One keypress from the real terminal. None on timeout."""
    import termios, tty
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ready, _, _ = select.select([fd], [], [], timeout)
        if not ready:
            return None
        return os.read(fd, 3).decode("utf-8", errors="ignore")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


PANEL = (
    "\n\033[36m╭─ Rate this response ───────────╮\033[0m\n"
    "\033[36m│\033[0m  Rate good AND bad for best    \033[36m│\033[0m\n"
    "\033[36m│\033[0m  results                       \033[36m│\033[0m\n"
    "\033[36m│\033[0m                                \033[36m│\033[0m\n"
    "\033[36m│\033[0m  \033[32m[1] Good\033[0m  \033[31m[2] Bad\033[0m  \033[90m[3] Skip\033[0m  \033[36m│\033[0m\n"
    "\033[36m│\033[0m  \033[90m[↑] Rate previous response\033[0m   \033[36m│\033[0m\n"
    "\033[36m╰────────────────────────────────╯\033[0m\n"
)

def _clear_panel():
    sys.stderr.write("\033[8A\033[J")
    sys.stderr.flush()


def collect_rating(timeout_s=30) -> Optional[int]:
    """Show panel, read one key. Returns +1, -1, or None."""
    fd = _open_tty()
    if fd is None:
        return None

    steps = max(1, int(timeout_s / 0.1))
    sys.stderr.write(PANEL)
    sys.stderr.flush()
    try:
        for _ in range(steps):
            key = _raw_read(fd, 0.1)
            if key is None:
                continue
            if key.startswith("\x1b"):
                _clear_panel()
                return None
            _clear_panel()
            return _KEYS.get(key)
    finally:
        os.close(fd)
    _clear_panel()
    return None
