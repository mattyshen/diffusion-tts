import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_path(log_dir: str, prefix: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return str(Path(log_dir) / f"{prefix}_{ts}.jsonl")


class SearchDataLogger:
    def __init__(self, path: str, run_meta: Optional[Dict[str, Any]] = None):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")
        if run_meta is not None:
            self.log({"event_type": "run_meta", "ts_utc": utc_now_iso(), "run_meta": run_meta})

    def log(self, event: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(event, ensure_ascii=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

