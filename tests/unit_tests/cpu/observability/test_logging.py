# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import logging
import sys

import pytest

from torchtitan.observability.logging import init_logger


@pytest.fixture
def restore_root_logger():
    root_logger = logging.getLogger()
    original_level = root_logger.level
    original_handlers = root_logger.handlers[:]
    yield root_logger
    root_logger.setLevel(original_level)
    root_logger.handlers[:] = original_handlers


def test_init_logger_configures_named_loggers(
    monkeypatch: pytest.MonkeyPatch, restore_root_logger: logging.Logger
) -> None:
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)

    init_logger()
    logger = logging.getLogger("torchtitan.example")
    logger.info("message")

    assert "torchtitan.example - INFO - message" in output.getvalue()


def test_init_logger_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, restore_root_logger: logging.Logger
) -> None:
    monkeypatch.setattr(sys, "stdout", io.StringIO())

    init_logger()
    init_logger()

    assert len(restore_root_logger.handlers) == 1


@pytest.mark.parametrize("rank,info_visible", [("0", True), ("7", False)])
def test_rank_filter_preserves_warnings_and_errors(
    monkeypatch, restore_root_logger, rank, info_visible
):
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setenv("RANK", rank)
    monkeypatch.setenv("LOG_RANK", "0")
    init_logger()
    logger = logging.getLogger("torchtitan.example")
    logger.info("training metrics")
    logger.warning("rank warning")
    logger.error("rank failure")
    text = output.getvalue()
    assert ("training metrics" in text) == info_visible
    assert "rank warning" in text and "rank failure" in text
    assert f"[rank {rank}]" in text


def test_explicit_global_log_ranks(monkeypatch, restore_root_logger):
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setenv("RANK", "8")
    monkeypatch.setenv("LOG_RANK", "0,8")
    init_logger()
    logging.getLogger("torchtitan.example").info("node diagnostic")
    assert "node diagnostic" in output.getvalue()


def test_unset_log_rank_preserves_native_console_behavior(
    monkeypatch, restore_root_logger
):
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setenv("RANK", "7")
    monkeypatch.delenv("LOG_RANK", raising=False)
    init_logger()
    logging.getLogger("torchtitan.example").info("native rank output")
    assert "native rank output" in output.getvalue()
