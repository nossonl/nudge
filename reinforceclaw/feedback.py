"""Terminal feedback panel. Shows Good/Bad/Skip after each response."""
# reads keypresses directly from the terminal, not stdin
# blocks until you press 1/2/3 or any other key to skip

import os
import sys
from typing import Optional

_KEYS = {"1": 1, "2": -1, "3": None}


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
    "\033[32m╰────────────────────────────────╯\033[0m\n"
)


def _clear_panel():
    sys.stderr.write("\033[7A\033[J")  # 7 lines now (removed arrow row)
    sys.stderr.flush()


def collect_rating() -> Optional[int]:
    """Show panel, wait for keypress. Returns +1, -1, or None."""
    fd = _open_tty()
    if fd is None:
        sys.stderr.write("reinforceclaw: panel unavailable; use /good or /bad.\n")
        sys.stderr.flush()
        return None

    sys.stderr.write(PANEL)
    sys.stderr.flush()

    import termios, tty
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = os.read(fd, 3).decode("utf-8", errors="ignore")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        os.close(fd)

    _clear_panel()

    if not ch or ch.startswith("\x1b"):
        return None
    return _KEYS.get(ch)
