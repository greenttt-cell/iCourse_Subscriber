"""Small tasks for exercising the actual spawned-worker lifecycle."""

import subprocess
import sys
import time
from pathlib import Path


class EmptyPPTVPN:
    """Pickleable VPN stand-in for an already-transcribed lecture."""

    def get(self, url, **kwargs):
        from types import SimpleNamespace
        if not url.endswith("/pptnote/v1/schedule/search-ppt"):
            raise AssertionError(f"Unexpected API request: {url}")
        return SimpleNamespace(json=lambda: {"code": 0, "list": []},
                               raise_for_status=lambda: None)


def save_transcript(db_path, delay=0):
    from src.data.database import Database
    db = Database(db_path)
    db.upsert_course("course", "Test course", "")
    db.insert_lecture("lecture", "course", "Test lecture", "2026-10-01")
    db.update_transcript("lecture", "committed transcript")
    time.sleep(delay)
    db.conn.close()


def spawn_stuck_download(pid_path):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    Path(pid_path).write_text(str(child.pid))
    time.sleep(60)


def fail():
    raise RuntimeError("simulated worker failure")


def process_with_local_contract(vpn, db_path, course_id, title, lecture, base_url):
    """Trust only the explicit loopback response fixture in this spawned test."""
    from urllib.parse import urlsplit
    from src.runtime import config
    import main

    endpoint = urlsplit(base_url)
    if endpoint.scheme != "http" or endpoint.hostname != "127.0.0.1":
        raise AssertionError("Test worker must use the loopback fixture")
    config.MODEL_PROVIDER_ALLOWLIST = ("modelscope",)
    config.MODEL_RESPONSE_CONTRACTS = {
        ("modelscope", model): {
            "status": "verified", "adapter": "openai_chat_completions",
            "base_url": base_url, "source": "explicit local HTTP test fixture",
        }
        for model in ("deepseek-ai/DeepSeek-V4-Pro", "Qwen/Qwen3.5-397B-A17B")
    }
    main._process_lecture(vpn, db_path, course_id, title, lecture)
