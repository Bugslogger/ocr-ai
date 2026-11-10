#!/usr/bin/env python3
"""Run a command and save its combined console output to rotating log files."""
import argparse
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import signal
import subprocess
import sys


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="application", help="Log filename prefix")
    parser.add_argument("--log-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "logs")
    parser.add_argument("--max-bytes", type=positive_integer, default=10 * 1024 * 1024)
    parser.add_argument("--backup-count", type=positive_integer, default=5)
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="Command to run, preceded by --")
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("provide a command after --")
    if not args.name or args.name in {".", ".."} or any(c in args.name for c in "/\\"):
        parser.error("--name must be a filename prefix without path separators")

    args.log_dir = args.log_dir.expanduser().resolve()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("command_runner")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    file_handler = RotatingFileHandler(
        args.log_dir / f"{args.name}.log", maxBytes=args.max_bytes,
        backupCount=args.backup_count, encoding="utf-8",
    )
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler.setFormatter(formatter)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console)
    child = None
    previous_handlers = {}

    def forward_signal(signum, _frame):
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    try:
        # Python children flush promptly even when stdout is a pipe.
        environment = dict(os.environ, PYTHONUNBUFFERED="1")
        child = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            env=environment,
        )
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, forward_signal)
        logger.info("Started process pid=%s; log=%s", child.pid, file_handler.baseFilename)
        with child.stdout:
            for line in child.stdout:
                # Child severity/traceback text is preserved within the message.
                logger.info("%s", line.rstrip("\r\n"))
        code = child.wait()
        logger.log(logging.INFO if code == 0 else logging.ERROR,
                   "Process exited with status %s", code)
        return code if code >= 0 else 128 - code
    except OSError:
        logger.exception("Unable to run command")
        return 1
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        logger.removeHandler(file_handler)
        logger.removeHandler(console)
        file_handler.close()
        console.close()


if __name__ == "__main__":
    sys.exit(main())
