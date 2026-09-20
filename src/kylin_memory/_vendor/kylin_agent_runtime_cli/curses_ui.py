import sys

def flush_stdin() -> None:
    """Flush any stray bytes from the stdin input buffer.

    Must be called after ``curses.wrapper()`` (or any terminal-mode library
    like simple_term_menu) returns, **before** the next ``input()`` /
    ``getpass.getpass()`` call.  ``curses.endwin()`` restores the terminal
    but does NOT drain the OS input buffer — leftover escape-sequence bytes
    (from arrow keys, terminal mode-switch responses, or rapid keypresses)
    remain buffered and silently get consumed by the next ``input()`` call,
    corrupting user data (e.g. writing ``^[^[`` into .env files).

    On non-TTY stdin (piped, redirected) or Windows, this is a no-op.
    """
    try:
        if not sys.stdin.isatty():
            return
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass

