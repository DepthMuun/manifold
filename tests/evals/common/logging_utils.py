import json
import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Any, Dict
from datetime import datetime


def setup_logger(name: str, logfile: str, level: int = logging.INFO, max_mb: int = 5, backups: int = 3) -> logging.Logger:
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        fh = RotatingFileHandler(logfile, maxBytes=max_mb * 1024 * 1024, backupCount=backups)
        fh.setFormatter(fmt)
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


def json_log(logger: logging.Logger, event: str, payload: Dict[str, Any]):
    record = {
        "event": event,
        "ts": datetime.utcnow().isoformat() + "Z",
        **payload,
    }
    logger.info(json.dumps(record))
