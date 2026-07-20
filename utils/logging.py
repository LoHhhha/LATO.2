# copied and modified from https://github.com/LoHhhha/pmos_nn/blob/master/flowing/shower/Logger.py

import os
import inspect
import time

DEBUG_MSG = 0
INFO_MSG = 1
WARNING_MSG = 2
ERROR_MSG = 3
FAULT_MSG = 4
PACKAGE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOGGER_LEVER = INFO_MSG

MSG_TYPES_STR = [
    "\033[1m\033[36mD\033[0m",
    "\033[1m\033[32mI\033[0m",
    "\033[1m\033[33mW\033[0m",
    "\033[1m\033[35mE\033[0m",
    "\033[1m\033[31mF\033[0m",
]


def debug(*msg):
    __print_out(DEBUG_MSG, *msg)


def info(*msg):
    __print_out(INFO_MSG, *msg)


def warning(*msg):
    __print_out(WARNING_MSG, *msg)


def error(*msg):
    __print_out(ERROR_MSG, *msg)


def fault(*msg):
    __print_out(FAULT_MSG, *msg)


def __print_out(msg_type: int, *msg) -> None:
    if msg_type < LOGGER_LEVER:
        return

    # inspect.stack()[1] is info/warning/error
    caller_frame = inspect.stack()[2]

    caller_name = (
        os.path.relpath(caller_frame.filename, PACKAGE_PATH)
        .split(".")[0]
        .replace("\\", "/")
        .replace("/", ".")
    )

    print(
        f"[{MSG_TYPES_STR[msg_type]}|{time.strftime('%H:%M:%S')}|{caller_name}:{caller_frame.lineno}]",
        *msg,
    )


if __name__ == "__main__":
    debug("debug msg")
    info("info msg")
    warning("warning msg")
    error("error msg")
    fault("fault msg")
