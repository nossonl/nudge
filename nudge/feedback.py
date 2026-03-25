"""Terminal feedback panel. Shows Good/Bad/Skip after each response."""
# reads keypresses directly from the terminal, not stdin
# no timers — panel stays until you rate (1/2/3) or send your next prompt
# if you start typing normally the panel just goes away silently

import os
import sys
import select
from typing import Optional

_KEYS = {"1": 1, "2": -1, "3": None}  # good, bad, skip


def _open_tty():
    try:
        return os.open("/dev/tty", os.O_RDONLY)
    except OSError:
        return None


PANEL = (
    "\n\033[32m╭─ Rate this response ───────────╮\033[0m\n"
    "\033[32m│\033[0m  Rate good AND bad for best    \033[32m│\033[0m\n"
    "\033[32m│\033[0m  results                       \033[32m│\033[0m\n"
    "\033[32m│\033[0m                                \033[32m│\033[0m\n"
    "\033[32m│\033[0m  \033[32m[1] Good\033[0m  \033[31m[2] Bad\033[0m  \033[90m[3] Skip\033[0m  \033[32m│\033[0m\n"
    "\033[32m│\033[0m  \033[90m[↑] Rate previous response\033[0m   \033[32m│\033[0m\n"
    "\033[32m╰────────────────────────────────╯\033[0m\n"
)


def _clear_panel():
    sys.stderr.write("\033[8A\033[J")
    sys.stderr.flush()


def collect_rating() -> Optional[int]:
    """Show panel. If user presses 1/2/3, return rating. Anything else = skip."""
    fd = _open_tty()
    if fd is None:
        return None

    sys.stderr.write(PANEL)
    sys.stderr.flush()

    import termios, tty
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        # wait for any keypress — no timeout
        ch = os.read(fd, 3).decode("utf-8", errors="ignore")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        os.close(fd)

    _clear_panel()

    if not ch or ch.startswith("\x1b"):
        return None  # escape/arrow = skip
    return _KEYS.get(ch)  # 1=good, 2=bad, anything else=None (skip)
