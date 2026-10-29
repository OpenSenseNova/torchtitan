# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Console logging setup and helpers.

Library modules should create a named logger with ``logging.getLogger(__name__)``.
Process entry points should call :func:`init_logger` once before doing work.
"""

import logging
import os
import sys

__all__ = ["init_logger", "warn_once"]


class _RankFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.rank = int(os.environ.get("RANK", "0"))
        selected = os.environ.get("LOG_RANK")
        self.ranks = (
            {int(rank) for rank in selected.split(",") if rank.strip()}
            if selected is not None
            else None
        )

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = self.rank
        return (
            record.levelno >= logging.WARNING
            or self.ranks is None
            or self.rank in self.ranks
        )


def init_logger() -> None:
    """Optionally select global INFO ranks with LOG_RANK; keep warnings/errors everywhere."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.addFilter(_RankFilter())
    formatter = logging.Formatter(
        "[titan][rank %(rank)d] %(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # suppress verbose torch.profiler logging
    os.environ["KINETO_LOG_LEVEL"] = "5"


_logged: set[str] = set()


def warn_once(logger: logging.Logger, msg: str) -> None:
    """Log a warning message only once per unique message.

    Uses a global set to track messages that have already been logged
    to prevent duplicate warning messages from cluttering the output.

    Args:
        logger (logging.Logger): The logger instance to use for warning.
        msg (str): The warning message to log.
    """
    if msg not in _logged:
        logger.warning(msg)
        _logged.add(msg)
