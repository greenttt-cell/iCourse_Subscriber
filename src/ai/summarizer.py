"""LLM-based course lecture summarization via ModelScope API."""

import json
import math
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from openai import OpenAI

from src.runtime import config

SYSTEM_PROMPT= r"""你是一个专业的课程助教。你的任务是根据用户提供的课程录音文本和ppt文字ocr部分，生成用于学生自学和期末复习的详细课堂讲义，完整保留材料中的知识与论证。
1. **直接输出**：不要包含任何"好的"、"没问题"、"以下是总结"等客套话，不要输出全局课程名称大标题（由系统自动生成），直接开始总结即可。
2. **文本清洗**：语言必须通顺、逻辑清晰，可以合并重复口述、去除无意义的语气词和识别噪声，但不得删除有意义的推导步骤、条件说明或不同角度的解释。内容可能被识别成同音字，仅在上下文或课件支持时修复；无法确认的术语或公式应明确标注存疑。
3. **格式严格**：
   - 必须使用 Markdown 格式排版。
   - 标题结构清晰与级别限制：只允许使用三级及后续级别的标题（即只能使用`###`、`####`或`#####`），禁止使用 `#` 和 `##`。用清晰的标题组织结构。
   - 不得使用超过两级的缩进和超过两级嵌套的bullet point。
   - 尽可能用完整的段落来组织老师的讲解，适量使用bullet point，不得过度使用bullet point
   - 合理使用加粗、列表、表格、分级标题、段首小标题等形式来组织信息，确保结构清晰。
4. **公式规范**：所有数学公式或科学变量必须使用规范的 LaTeX 语法（行内公式用 $...$，行间公式用 $$...$$）。由于图床限制，latex公式中不要出现中文。
5. **忠于材料与内容完整**：以材料中实际包含的知识和论证决定篇幅，不设置固定字数、压缩比例或篇幅下限。不要为了简短省略关键内容，也不要为了增加长度重复内容。一般课程应保留核心概念、论证、案例及提及的文献；数学课程还应逐项保留以下内容：
   - **定义与记号**：写清对象、符号含义、定义域、量词、限定条件及适用范围。
   - **定理、引理与推论**：完整写出假设、结论、适用条件，以及材料中说明的相互关系，不能只保留定理名称或一句结论。
   - **证明与推导**：按讲解的逻辑展开材料中已有的证明过程和中间等式、不等式、变量替换、分类讨论及关键理由。不得把有意义的步骤压缩为“可得”“类似可证”或只给最终公式。
   - **例题与练习讲解**：保留完整题设、已给条件、求解目标、解题思路、实际讲解的步骤和最终结论，以及材料中出现的其他解法或适用条件讨论。
   - **观察、技巧与反例**：保留讲师给出的动机、关键观察、计算技巧、反例及易错点，并写清它们对应的问题或条件。
   - **材料缺失与不确定性**：仅整理输入材料支持的内容。若证明、计算或题设缺失，应在相应位置明确标注“原始材料未提供该步骤”“题设不完整”或具体存疑之处；不得自行补写缺失的数学推导、例题、结论或引用，不能用常识或模型记忆冒充课堂内容。
6. 你需要格外注意课程中是否提及了作业、考试、签到、组队等关键事项，如果有的话，用三级标题【课程事项提醒】标注在开头。
7. **文风与格式示例**：以下关于"梯度下降"的短片段仅示范段落组织、术语解释和公式排版，不代表整节课的篇幅或内容覆盖程度。实际输出应按上述要求保留材料中完整的定义、条件、证明、推导和例题；示例中的知识也不能作为输入材料之外的内容添加进讲义。

【❌ 错误的风格】
## 梯度下降

**定义：**
- 梯度下降是一种优化算法
- 用于最小化损失函数
- 广泛应用于机器学习

**核心步骤：**
- 计算梯度
- 更新参数
- 重复迭代

**学习率：**
- 学习率决定步长
- 太大会发散
- 太小会收敛慢
- 需要调参

**类型：**
- 批量梯度下降（BGD）
- 随机梯度下降（SGD）
- 小批量梯度下降（Mini-batch GD）

---

【✅ 正确的风格】

### 梯度下降

梯度下降是最小化损失函数 $L(\theta)$ 的核心优化算法。其基本思想是沿着损失函数对参数 $\theta$ 的梯度的反方向迭代更新，每一步的更新公式为 $\theta \leftarrow \theta - \eta \nabla_\theta L(\theta)$，其中 $\eta$ 称为学习率，控制每次更新的步长大小。

**学习率的选取至关重要**：若 $\eta$ 过大，参数更新幅度过猛，损失函数可能在最优点附近震荡甚至发散；若 $\eta$ 过小，收敛速度极慢，训练成本大幅上升。实践中通常通过**学习率调度（learning rate schedule）**或**自适应方法（如 Adam）**来缓解这一问题。

根据每次更新时使用的样本量，梯度下降可分为三类：**批量梯度下降（BGD）** 每次使用全部训练数据，梯度估计准确但计算开销大；**随机梯度下降（SGD）** 每次仅用单个样本，更新频繁但噪声大；**小批量梯度下降（Mini-batch GD）** 则折中两者，是深度学习中最常用的形式。
---

8. **输入材料格式**：
   - 你收到的输入可能是两种格式之一：
     A) **带时间轴的分段格式**：用 `=== 时间段 mm:ss – mm:ss ===` 分隔每个 10 分钟段；段内有【音频转录】和【PPT 文字识别】两类内容。
     B) **平铺格式**：先一段【音频转录（无时间轴）】，后一段【PPT 文字识别（按出现顺序）】，每张 PPT 带 `[页 N @ mm:ss]` 标签。
   - 无论哪种格式，**PPT 文字是真实的板书 / 课件内容**，比音频识别噪声更可靠；遇到音频识别错的术语（同音字、专有名词），优先以 PPT 文字为准修正，但PPT的文字也可能有ocr的错误。
   - 但是，你的总结的组织的主线仍然应该以讲师的讲解为逻辑组织，同时把PPT的信息补充进去。
   - 同时，需要注意的是，由于该平台以屏幕截图的方式记录ppt，因此，尽管我们提供的ocr版本已经尽最大努力进行了清洗，你收到的ppt仍然可能有无关网页、导航栏、桌面或系统页面等无关噪音。这些噪音不应被用于课程内容的总结。
   - 输出仍按之前的格式要求，不要保留时间戳标签，只把这些信息当作上下文辅助理解，不用说哪些来自转写哪些来自ppt，自然地合并录音转写和ppt中的知识，生成高质量笔记。"""


@dataclass(frozen=True)
class SummaryResult:
    """Receipt whose verified normal ending permits publication.

    contract_status covers the normal finish rule, not output budget support
    or the hosted model's entire API contract; diagnostics keep those separate.
    """

    summary: str
    model_used: str
    completion_status: str
    finish_reason: Any
    contract_status: str
    usage: dict | None = None
    provider: str | None = None
    requested_model: str | None = None
    response_model: str | None = None
    elapsed_seconds: float | None = None
    diagnostics: dict | None = None

    @property
    def is_complete(self) -> bool:
        return (
            self.completion_status == "complete"
            and self.contract_status == "verified"
            and isinstance(self.summary, str)
            and bool(self.summary.strip())
            and self.finish_reason == "stop"
        )


class SummaryGenerationError(RuntimeError):
    """All allowed attempts failed, with metadata but no source/answer text."""

    def __init__(self, attempts: list[dict]):
        self.attempts = list(attempts)
        super().__init__("No verified complete summary from allowed models")


class SummaryAttemptError(RuntimeError):
    def __init__(self, metadata: dict):
        self.metadata = metadata
        super().__init__("LLM attempt failed: " + metadata.get("error_type", "error"))


def _field(value: Any, name: str, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _metadata_value(value: Any):
    """Convert metadata only, never arbitrary response objects or their repr."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _metadata_value(v) for k, v in value.items()}
    if isinstance(value, SimpleNamespace):
        return _metadata_value(vars(value))
    if isinstance(value, (list, tuple)):
        return [_metadata_value(v) for v in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        result = dump()
        return _metadata_value(result) if isinstance(result, dict) else None
    return None


def _redact_metadata_text(value: Any):
    """Keep numeric diagnostic structure without logging provider text."""
    if isinstance(value, str):
        return "[redacted]"
    if isinstance(value, dict):
        return {key: _redact_metadata_text(item) for key, item in value.items()
                if key != "reasoning_content"}
    if isinstance(value, list):
        return [_redact_metadata_text(item) for item in value]
    return value


def _usage_dict(value: Any) -> dict | None:
    if value is None:
        return None
    plain = _metadata_value(value)
    if isinstance(plain, dict):
        return plain
    # Also support simple provider/test objects without serializing anything
    # other than known token-count fields.
    result = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        count = _field(value, name)
        if isinstance(count, int) and not isinstance(count, bool):
            result[name] = count
    details = _field(value, "completion_tokens_details")
    for name in ("reasoning_tokens", "text_tokens"):
        count = _field(details, name)
        if isinstance(count, int) and not isinstance(count, bool):
            result.setdefault("completion_tokens_details", {})[name] = count
        top_count = _field(value, name)
        if isinstance(top_count, int) and not isinstance(top_count, bool):
            result[name] = top_count
    return result or None


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _same_endpoint(actual: Any, expected: Any) -> bool:
    return (
        isinstance(actual, str) and isinstance(expected, str)
        and bool(actual) and actual.rstrip("/") == expected.rstrip("/")
    )


class OpenAIChatCompletionAdapter:
    """Interpret this response shape without assuming all gateways share it."""

    ABNORMAL_REASONS = frozenset({
        "content_filter", "refusal", "tool_calls", "function_call", "aborted",
        "insufficient_system_resource", "error", "cancelled", "canceled",
    })

    @classmethod
    def inspect(cls, response: Any, contract_status: str) -> tuple[str, Any, dict | None, str]:
        choices = _field(response, "choices")
        choice = choices[0] if isinstance(choices, (list, tuple)) and choices else None
        message = _field(choice, "message")
        # reasoning_content is private model output, never lecture content.
        text = _field(message, "content")
        text = text.strip() if isinstance(text, str) else ""
        reason = _field(choice, "finish_reason")
        usage = _usage_dict(_field(response, "usage"))
        if choice is None:
            # Native Gemini responses at the configured OpenAI route are not
            # silently interpreted as Chat Completions. Keep only metadata.
            candidates = _field(response, "candidates")
            if isinstance(candidates, (list, tuple)) and candidates:
                reason = _field(candidates[0], "finishReason")
                usage = _usage_dict(_field(response, "usageMetadata"))
            return text, reason, usage, "unknown"
        if _field(message, "refusal") or _field(message, "tool_calls") or _field(message, "function_call"):
            status = "abnormal"
        elif reason == "length":
            status = "truncated"
        elif isinstance(reason, str) and reason in cls.ABNORMAL_REASONS:
            status = "abnormal"
        elif reason == "stop" and contract_status == "verified":
            status = "complete" if text else "abnormal"
        else:
            status = "unknown"
        return text, reason, usage, status


class Summarizer:
    """Try each explicitly allowed provider/model at most once, in config order."""

    def __init__(self):
        self.providers = config.resolve_model_providers()
        if not self.providers:
            raise ValueError("No explicitly allowed model provider has an API key")
        self._clients = {
            p["name"]: OpenAI(api_key=p["api_key"], base_url=p["base_url"], max_retries=0)
            for p in self.providers
        }

    @staticmethod
    def _contract(provider_name: str, model: str, base_url: str | None) -> tuple[str, str]:
        policy = config.MODEL_RESPONSE_CONTRACTS.get((provider_name, model), {})
        adapter = policy.get("adapter", "unrecognized")
        # A narrowly observed stop rule is enough for this publication gate.
        # Do not promote the model's remaining contract or its output budget.
        observed_stop = (
            policy.get("status") == "partially_verified"
            and policy.get("normal_finish_reason_status") == "observed"
            and policy.get("normal_finish_reasons") == ("stop",)
        )
        verified = (
            (policy.get("status") == "verified" or observed_stop)
            and adapter == "openai_chat_completions"
            and _same_endpoint(base_url, policy.get("base_url"))
        )
        return ("verified" if verified else "needs_verification"), adapter

    @staticmethod
    def _budget(provider_name: str, model: str, base_url: str | None) -> tuple[dict, dict]:
        policy = config.MODEL_OUTPUT_BUDGETS.get((provider_name, model), {})
        value, parameter = policy.get("value"), policy.get("parameter")
        parameter_status = policy.get("parameter_status", "needs_verification")
        if not _same_endpoint(base_url, policy.get("base_url")):
            parameter_status = "needs_verification"
        metadata = {
            "budget_status": "budget_unconfigured", "budget_parameter": parameter,
            "requested_budget": value, "budget_parameter_status": parameter_status,
        }
        if value is None:
            return {}, metadata
        if not (
            parameter_status == "verified"
            and parameter in ("max_tokens", "max_completion_tokens")
            and _same_endpoint(base_url, policy.get("base_url"))
            and isinstance(value, int) and not isinstance(value, bool) and value > 0
        ):
            raise ValueError("Output budget requires a verified endpoint/parameter and positive integer")
        metadata["budget_status"] = "budget_configured"
        return {parameter: value}, metadata

    @staticmethod
    def _record(metadata: dict) -> None:
        try:
            # Work on a fresh JSON-compatible copy: receipts and exceptions
            # retain the original metadata, including provider extensions.
            record = _metadata_value(metadata)
            record["raw_usage"] = _redact_metadata_text(record.get("raw_usage"))
            reason = record.get("raw_finish_reason")
            short_enum = (
                isinstance(reason, str) and len(reason) <= 64
                and all(char.isascii() and (char.isalnum() or char == "_") for char in reason)
            )
            if not short_enum:
                record["raw_finish_reason"] = _redact_metadata_text(reason)
            record["log_metadata_redacted"] = True
            line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        except Exception:
            # Diagnostics must not bypass the bounded fallback path, and an
            # encoding failure must never print the object or exception text.
            line = '{"event":"llm_attempt","log_metadata_redacted":true,"log_status":"serialization_failed"}'
        print("[Summarizer] " + line)

    def _call_llm(self, client: OpenAI, model: str, title: str, content: str,
                  *, provider_name: str, base_url: str | None = None) -> SummaryResult:
        t0 = time.monotonic()
        if base_url is None:
            candidate = getattr(client, "base_url", None)
            base_url = str(candidate) if candidate is not None else None
        contract_status, adapter = self._contract(provider_name, model, base_url)
        metadata = {
            "event": "llm_attempt", "provider": provider_name, "requested_model": model,
            "response_model": None, "elapsed_seconds": None, "raw_finish_reason": None,
            "raw_usage": None, "prompt_tokens": None, "completion_tokens": None,
            "total_tokens": None, "reasoning_tokens": None, "text_tokens": None,
            "classification": "unknown",
            "contract_status": contract_status, "response_adapter": adapter,
            "contract_scope": "normal_finish_reason",
            "budget_status": "budget_unconfigured", "budget_parameter": None,
            "requested_budget": None, "budget_parameter_status": "needs_verification",
            "request_sent": False,
        }
        response_policy = config.MODEL_RESPONSE_CONTRACTS.get((provider_name, model), {})
        matching_endpoint = _same_endpoint(base_url, response_policy.get("base_url"))
        metadata["model_contract_status"] = (
            response_policy.get("status", "needs_verification")
            if matching_endpoint else "needs_verification"
        )
        for name in ("normal_finish_reason_status", "truncation_finish_reason_status"):
            metadata[name] = (
                response_policy.get(name, "verified" if response_policy.get("status") == "verified"
                                    else "needs_verification")
                if matching_endpoint else "needs_verification"
            )
        budget_policy = config.MODEL_OUTPUT_BUDGETS.get((provider_name, model), {})
        metadata.update(
            budget_parameter=budget_policy.get("parameter"),
            requested_budget=budget_policy.get("value"),
            budget_parameter_status=budget_policy.get("parameter_status", "needs_verification"),
        )
        if not _same_endpoint(base_url, budget_policy.get("base_url")):
            metadata["budget_parameter_status"] = "needs_verification"
        response = None
        try:
            budget_kwargs, budget_metadata = self._budget(provider_name, model, base_url)
            metadata.update(budget_metadata)
            metadata["request_sent"] = True
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"以下是课程《{title}》的录音转录与课件文字材料。请根据材料生成可用于复习的详细课堂讲义，保留有意义的定义、条件、论证、推导和例题；篇幅由内容决定，不要重复凑字或补写缺失步骤：\n\n{content}",
                    },
                ],
                timeout=config.LLM_TIMEOUT_SECONDS,
                **budget_kwargs,
            )
            text, reason, usage, status = OpenAIChatCompletionAdapter.inspect(response, contract_status)
            if not _field(response, "choices") and _field(response, "candidates"):
                contract_status = "needs_verification"
                metadata["contract_status"] = contract_status
                metadata["response_shape"] = "unexpected_native_gemini"
            metadata["classification"] = status
            self._fill_metadata(metadata, response, reason, usage)
        except Exception as exc:
            # Error messages and HTTP bodies can echo prompts or credentials.
            # Extract only the same response metadata fields as the success path.
            source = response if response is not None else getattr(exc, "body", None)
            if not isinstance(source, dict) and response is None:
                http_response = getattr(exc, "response", None)
                json_method = getattr(http_response, "json", None)
                if callable(json_method):
                    try:
                        source = json_method()
                    except Exception:
                        source = None
            try:
                _, reason, usage, _ = OpenAIChatCompletionAdapter.inspect(source, contract_status)
                self._fill_metadata(metadata, source, reason, usage)
            except Exception:
                pass
            metadata.update(classification="abnormal", error_type=type(exc).__name__)
            if not metadata["request_sent"] and metadata["requested_budget"] is not None:
                metadata["budget_status"] = "budget_invalid"
            status_code = getattr(exc, "status_code", None)
            if isinstance(status_code, int):
                metadata["http_status"] = status_code
            metadata["elapsed_seconds"] = round(time.monotonic() - t0, 3)
            self._record(metadata)
            raise SummaryAttemptError(metadata) from None
        metadata["elapsed_seconds"] = round(time.monotonic() - t0, 3)
        self._record(metadata)
        return SummaryResult(
            summary=text, model_used=f"{provider_name}/{model}", completion_status=status,
            finish_reason=reason, contract_status=contract_status, usage=usage,
            provider=provider_name, requested_model=model, response_model=metadata["response_model"],
            elapsed_seconds=metadata["elapsed_seconds"], diagnostics=metadata,
        )

    @staticmethod
    def _fill_metadata(metadata: dict, response: Any, reason: Any, usage: dict | None) -> None:
        response_model = _field(response, "model")
        metadata["response_model"] = response_model if isinstance(response_model, str) else None
        metadata["raw_finish_reason"] = _metadata_value(reason)
        metadata["raw_usage"] = usage
        usage_fields = usage or {}
        if not _field(response, "choices") and _field(response, "candidates"):
            # These counters have native Gemini meanings.  In particular,
            # candidates and thoughts are recorded separately, never summed
            # into an invented OpenAI-style completion/total count.
            metadata["usage_counter_schema"] = "gemini_native_unverified"
            for target, source in (
                ("prompt_tokens", "promptTokenCount"),
                ("candidate_tokens", "candidatesTokenCount"),
                ("total_tokens", "totalTokenCount"),
                ("reasoning_tokens", "thoughtsTokenCount"),
            ):
                metadata[target] = _count(usage_fields.get(source))
            return
        metadata["usage_counter_schema"] = "openai_chat_completions"
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            metadata[name] = _count(usage_fields.get(name))
        # Provider details may overlap completion_tokens or one another.
        # Record their actual values, never sum them into a billing estimate.
        for name in ("reasoning_tokens", "text_tokens"):
            nested_count = _count(_field(usage_fields.get("completion_tokens_details"), name))
            metadata[name] = (
                nested_count if nested_count is not None else _count(usage_fields.get(name))
            )

    def summarize(self, title: str, content: str) -> SummaryResult:
        """Return a verified complete receipt, or raise with attempt diagnostics."""
        if not isinstance(content, str) or not content.strip():
            raise SummaryGenerationError([])
        attempts = []
        seen = set()
        for provider in self.providers:
            # Also enforce the allowlist at dispatch time for manually supplied
            # runtime configs; possessing a client/key is not authorization.
            if provider["name"] not in config.MODEL_PROVIDER_ALLOWLIST:
                continue
            client = self._clients[provider["name"]]
            for model in provider["models"]:
                key = (provider["name"], model)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    result = self._call_llm(
                        client, model, title, content, provider_name=provider["name"],
                        base_url=provider["base_url"],
                    )
                except SummaryAttemptError as exc:
                    attempts.append(exc.metadata)
                    continue
                attempts.append(result.diagnostics or {})
                if result.is_complete:
                    return result
        raise SummaryGenerationError(attempts)
