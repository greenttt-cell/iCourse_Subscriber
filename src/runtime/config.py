import os

STUDENT_ID = os.environ.get("StuId", "")
PASSWORD = os.environ.get("UISPsw", "")

WEBVPN_BASE = "https://webvpn.fudan.edu.cn"
IDP_BASE = "https://id.fudan.edu.cn"
ICOURSE_BASE = "https://icourse.fudan.edu.cn"

WEBVPN_AES_KEY = b"wrdvpnisthebest!"
WEBVPN_AES_IV = b"wrdvpnisthebest!"

TENANT_CODE = "222"
GROUP_CODE = "2095000001"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# 模型服务商配置（按列表顺序作为优先级，从前往后尝试）。
# 用户可以在这里随意添加/删除/重排服务商和模型。
# 兼容性：只设置 DASHSCOPE_API_KEY 也能跑（modelscope 项的 api_key 直接读取它）。
# 同名 provider 多次出现 → resolve_model_providers() 把它们的 models 合并到首次出
# 现的那条；这避免 Summarizer 内部按 name 索引 client 字典时被后写覆盖。
MODEL_PROVIDERS: list[dict] = [
    {
        "name": "modelscope",
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url_env": "DASHSCOPE_BASE_URL",
        "default_base_url": "https://api-inference.modelscope.cn/v1/",
        "models": [
            "deepseek-ai/DeepSeek-V4-Pro",
            "Qwen/Qwen3.5-397B-A17B",
        ],
    },
    {
        "name": "deepseek",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "default_base_url": "https://api.deepseek.com/v1",
        "models": [
            "deepseek-v4-flash"
        ],
    },
    # {
    #     "name": "modelscope",
    #     "api_key_env": "DASHSCOPE_API_KEY",
    #     "base_url_env": "DASHSCOPE_BASE_URL",
    #     "default_base_url": "https://api-inference.modelscope.cn/v1/",
    #     "models": [
    #         "deepseek-ai/DeepSeek-V3.2",
    #         "ZhipuAI/GLM-5",
    #         "MiniMax/MiniMax-M2.5",
    #         "Qwen/Qwen3.5-397B-A17B",
    #     ],
    # },
    {
        "name": "gemini",
        "api_key_env": "GEMINI_API_KEY",
        "base_url_env": "GEMINI_BASE_URL",
        "default_base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": [
            "gemini-2.5-flash",
            "gemini-3-flash-preview",
        ],
    }
]


# An API key alone does not authorize another provider or a paid fallback.
# This list filters MODEL_PROVIDERS without changing its declared order.
MODEL_PROVIDER_ALLOWLIST = tuple(
    name.strip()
    for name in os.environ.get("LLM_ALLOWED_PROVIDERS", "modelscope").split(",")
    if name.strip()
)

# Response contracts are specific to both the provider/model and endpoint.
# OpenAI SDK compatibility alone does not verify a provider's stop semantics.
MODEL_RESPONSE_CONTRACTS: dict[tuple[str, str], dict] = {
    ("modelscope", "deepseek-ai/DeepSeek-V4-Pro"): {
        "status": "needs_verification", "adapter": "openai_chat_completions",
        "base_url": "https://api-inference.modelscope.cn/v1/",
    },
    ("modelscope", "Qwen/Qwen3.5-397B-A17B"): {
        "status": "partially_verified", "adapter": "openai_chat_completions",
        "base_url": "https://api-inference.modelscope.cn/v1/",
        # Phase 2.1 probe 1: HTTP 200, message.content="OK", finish_reason="stop".
        # This verifies this normal ending only, not every hosted API behavior.
        "normal_finish_reasons": ("stop",),
        "normal_finish_reason_status": "observed",
        # Keep rejecting length defensively; it has not been observed here.
        "truncation_finish_reason_status": "needs_verification",
    },
    ("deepseek", "deepseek-v4-flash"): {
        "status": "verified", "adapter": "openai_chat_completions",
        "base_url": "https://api.deepseek.com/v1",
        "source": "https://api-docs.deepseek.com/api/create-chat-completion/",
    },
    ("gemini", "gemini-2.5-flash"): {
        "status": "needs_verification", "adapter": "openai_chat_completions",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
    ("gemini", "gemini-3-flash-preview"): {
        "status": "needs_verification", "adapter": "openai_chat_completions",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
}

# No production output budget is enabled before provider validation.  A
# documented parameter name does not establish a hosted model's token cap.
MODEL_OUTPUT_BUDGETS: dict[tuple[str, str], dict] = {
    ("modelscope", "deepseek-ai/DeepSeek-V4-Pro"): {
        "parameter": None, "parameter_status": "needs_verification", "value": None,
    },
    ("modelscope", "Qwen/Qwen3.5-397B-A17B"): {
        # Probe 2 had no HTTP response after 90.121s. Neither support nor
        # rejection of max_tokens=32 was established; send no budget parameter.
        "parameter": None, "parameter_status": "needs_verification", "value": None,
    },
    ("deepseek", "deepseek-v4-flash"): {
        "parameter": "max_tokens", "parameter_status": "verified", "value": None,
        "base_url": "https://api.deepseek.com/v1",
    },
    ("gemini", "gemini-2.5-flash"): {
        "parameter": None, "parameter_status": "needs_verification", "value": None,
    },
    ("gemini", "gemini-3-flash-preview"): {
        "parameter": None, "parameter_status": "needs_verification", "value": None,
    },
}


def resolve_model_providers() -> list[dict]:
    """Resolve MODEL_PROVIDERS into runtime configs.

    Filters the explicit allowlist and drops unset API keys. Same-name entries get
    their model lists merged into the first occurrence (Summarizer's client
    dict keys on name and would otherwise collide).

    Returns:
        list of {name, api_key, base_url, models}.
    """
    resolved: list[dict] = []
    by_name: dict[str, dict] = {}
    for p in MODEL_PROVIDERS:
        if p["name"] not in MODEL_PROVIDER_ALLOWLIST:
            continue
        api_key = os.environ.get(p["api_key_env"], "").strip()
        if not api_key:
            continue
        base_url = (
            os.environ.get(p.get("base_url_env", ""), "").strip()
            or p.get("default_base_url", "")
        )
        if not base_url:
            continue
        if p["name"] in by_name:
            existing = by_name[p["name"]]
            for m in p["models"]:
                if m not in existing["models"]:
                    existing["models"].append(m)
            continue
        entry = {
            "name": p["name"],
            "api_key": api_key,
            "base_url": base_url,
            "models": list(p["models"]),
        }
        resolved.append(entry)
        by_name[p["name"]] = entry
    return resolved


# Legacy compatibility shims (kept so other modules importing these don't break)
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# QQ SMTP
SMTP_EMAIL = os.environ.get("SMTP_EMAIL", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
RECEIVER_EMAIL = os.environ.get("RECEIVER_EMAIL", "")
SMTP_HOST = "smtp.qq.com"
SMTP_PORT = 465

# Database & Storage
DATA_DIR = os.environ.get("DATA_DIR", "data")
VIDEO_DIR = os.path.join(DATA_DIR, "videos")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")  # ffmpeg-decoded f32le scratch buffers
DB_PATH = os.environ.get("DB_PATH", os.path.join(DATA_DIR, "icourse.db"))

# Sherpa-onnx ASR model directory.  Default: SenseVoice (zh+en+ja+ko+yue, int8).
# ASR_MODEL_DIR is the new name; SENSEVOICE_MODEL_DIR is the legacy env var
# kept as a fallback so existing CI cache keys keep working.
ASR_MODEL_DIR = os.environ.get(
    "ASR_MODEL_DIR",
    os.environ.get(
        "SENSEVOICE_MODEL_DIR",
        "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
    ),
)
SENSEVOICE_MODEL_DIR = ASR_MODEL_DIR  # alias for any straggler imports
SILERO_VAD_PATH = os.environ.get("SILERO_VAD_PATH", "silero_vad.onnx")

# ASR backend selector — Transcriber dispatches on this.  When changing,
# ASR_MODEL_DIR must point at a matching sherpa-onnx model bundle:
#   sensevoice — sherpa-onnx-sense-voice-* (multi-lang CTC, single model)
#   firered    — sherpa-onnx-fire-red-asr2-ctc-* (CTC, single model.onnx)
#   zipformer  — sherpa-onnx-zipformer-* (transducer, encoder/decoder/joiner)
ASR_BACKEND = os.environ.get("ASR_BACKEND", "sensevoice").strip().lower()
# Inference thread count.  4 fully saturates a 4-vCPU GitHub runner.
ASR_NUM_THREADS = int(os.environ.get("ASR_NUM_THREADS", "4"))

# ── Scheduler concurrency knobs.  All overridable via env. ────────────────
# image_pool: image downloads are tiny and IO-bound, 20 saturates bandwidth
# without hammering the iCourse server.
IMAGE_WORKERS = int(os.environ.get("IMAGE_WORKERS", "20"))
# OCR pool: pool size is the hard ceiling; a fixed BoundedSemaphore(OCR_MAX_TARGET)
# gates live concurrency since RapidOCR is single-threaded CPU-bound.
OCR_MAX_WORKERS = int(os.environ.get("OCR_MAX_WORKERS", "8"))
# Fixed cap — no dynamic CPU-based adjustment.  RapidOCR is single-threaded;
# more than 2 concurrent workers don't increase throughput on 4-core runners.
OCR_MAX_TARGET = int(os.environ.get("OCR_MAX_TARGET", "2"))
# Two concurrent ffmpeg audio extractions: the current lecture being
# transcribed + one pre-decoded for the next lecture.  Bandwidth-fair sharing
# at 20 MB/s split = ~10 MB/s each.
VIDEO_DOWNLOAD_CONCURRENCY = int(
    os.environ.get("VIDEO_DOWNLOAD_CONCURRENCY", "2")
)

# Bound individual lectures, including native ASR/OCR and cleanup. Timed-out
# workers are stopped by the parent process; completed DB writes survive.
LECTURE_TIMEOUT_SECONDS = int(os.environ.get("LECTURE_TIMEOUT_SECONDS", "1800"))
RUN_BUDGET_SECONDS = int(os.environ.get("RUN_BUDGET_SECONDS", "19200"))
PPT_WAIT_TIMEOUT_SECONDS = int(os.environ.get("PPT_WAIT_TIMEOUT_SECONDS", "300"))
PPT_FETCH_TIMEOUT_SECONDS = int(os.environ.get("PPT_FETCH_TIMEOUT_SECONDS", "120"))
PPT_MAX_PAGES = int(os.environ.get("PPT_MAX_PAGES", "100"))
MAX_LECTURE_ERRORS = int(os.environ.get("MAX_LECTURE_ERRORS", "3"))
LECTURE_RETRY_COOLDOWN_SECONDS = int(os.environ.get("LECTURE_RETRY_COOLDOWN_SECONDS", "86400"))
LLM_TIMEOUT_SECONDS = int(os.environ.get("LLM_TIMEOUT_SECONDS", "180"))
ASR_IDLE_TIMEOUT_SECONDS = int(os.environ.get("ASR_IDLE_TIMEOUT_SECONDS", "120"))

# 是否优先使用 iCourse 官方字幕（跳过 ASR 转录）。默认关闭。
USE_OFFICIAL_TRANSCRIPT = (
    os.environ.get("USE_OFFICIAL_TRANSCRIPT", "").strip().lower()
    in ("1", "true", "yes")
)

# 监控的课程 ID 列表
COURSE_IDS = [
    c.strip()
    for c in os.environ.get("COURSE_IDS", "").split(",")
    if c.strip()
]

# 学期级课程目录爬取（已弃用 — main.py 现在自动发现所有学期）。
# 保留此变量仅用于兼容老部署环境，新部署无需设置。
# 例：CRAWL_TERM=25
CRAWL_TERM = os.environ.get("CRAWL_TERM", "").strip()
