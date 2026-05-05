"""Terminal feedback panel. Shows Good/Bad/Skip after each response."""
# reads keypresses directly from the terminal, not stdin
# blocks until you press 1/2/3 or any other key to skip

import os
import sys
from typing import Literal

UNAVAILABLE = "unavailable"

_KEYS = {"1": 1, "2": -1, "3": None}


PANEL = (
    "\n\033[32m╭─ Rate this response ───────────╮\033[0m\n"
    "\033[32m│\033[0m  Rate good AND bad for best    \033[32m│\033[0m\n"
    "\033[32m│\033[0m  results                       \033[32m│\033[0m\n"
    "\033[32m│\033[0m                                \033[32m│\033[0m\n"
    "\033[32m│\033[0m  \033[32m[1] Good\033[0m  \033[31m[2] Bad\033[0m  \033[90m[3] Skip\033[0m  \033[32m│\033[0m\n"
    "\033[32m╰────────────────────────────────╯\033[0m\n"
)


def _clear_panel():
    sys.stderr.write(f"\033[{PANEL.count(chr(10))}A\033[J")
    sys.stderr.flush()


def collect_rating() -> int | None | Literal["unavailable"]:
    """Show panel, wait for keypress. Returns +1, -1, None, or unavailable."""
    try:
        fd = os.open("/dev/tty", os.O_RDONLY)
    except OSError:
        sys.stderr.write("reinforceclaw: panel unavailable; use /rl good or /rl bad.\n")
        sys.stderr.flush()
        return UNAVAILABLE

    sys.stderr.write(PANEL)
    sys.stderr.flush()

    import termios, tty
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = os.read(fd, 8).decode("utf-8", errors="ignore")
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        finally:
            os.close(fd)

    _clear_panel()

    if not ch or ch.startswith("\x1b"):
        return None
    return _KEYS.get(ch[:1])
