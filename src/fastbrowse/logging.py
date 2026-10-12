from __future__ import annotations

import logging

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``fastbrowse`` namespace."""
    return logging.getLogger(f"fastbrowse.{name}")


def setup_logging(*, level: int = logging.INFO) -> None:
    """Configure a console handler for the ``fastbrowse`` loggers (idempotent)."""
    logger = logging.getLogger("fastbrowse")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
    logger.propagate = False


def log_decision(logger: logging.Logger, tag: str, request: dict, response) -> None:
    """Log a full decision payload (questions + response) at INFO."""
    logger.info(
        "%s decision | question=%s | response=%s",
        tag,
        {qid: q.model_dump() for qid, q in request["questions"].items()},
        response.model_dump(),
    )
