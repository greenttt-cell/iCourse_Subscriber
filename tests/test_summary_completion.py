"""Local response fixtures for completion gates, budgets and usage metadata."""

import contextlib
import copy
import io
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.ai.summarizer import (
    Summarizer, SummaryAttemptError, SummaryGenerationError, SummaryResult,
)
from src.runtime import config


MISSING = object()
PRODUCTION_CONTRACTS = copy.deepcopy(config.MODEL_RESPONSE_CONTRACTS)
PRODUCTION_BUDGETS = copy.deepcopy(config.MODEL_OUTPUT_BUDGETS)
BASE_URL = "https://example.invalid/v1"
MODELSCOPE_URL = "https://api-inference.modelscope.cn/v1/"
MODELSCOPE_MODEL = "Qwen/Qwen3.5-397B-A17B"
PRIVATE_REASONING = "PRIVATE-REASONING-NOT-FOR-NOTES-OR-LOGS"


def response(reason="stop", text="A complete supplied proof.", usage=None):
    choice = {"message": {"content": text}, "index": 0}
    if reason is not MISSING:
        choice["finish_reason"] = reason
    return {"model": "returned-model", "choices": [choice], "usage": usage}


def observed_modelscope_response():
    # Counts/content/finish_reason from the user-supplied Phase 2.1 probe.
    # The reasoning text was not supplied; use a private synthetic sentinel.
    return {
        "model": MODELSCOPE_MODEL,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": "OK", "reasoning_content": PRIVATE_REASONING,
        }}],
        "usage": {"prompt_tokens": 14, "completion_tokens": 227, "total_tokens": 241,
                  "completion_tokens_details": {"reasoning_tokens": 221, "text_tokens": 227}},
    }


def logs(output):
    return [json.loads(line[len("[Summarizer] "):])
            for line in output.splitlines() if line.startswith("[Summarizer] {")]


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.providers = [{"name": "test", "api_key": "test-key",
                           "base_url": BASE_URL, "models": ["first", "backup"]}]
        contracts = {("test", model): {
            "status": "verified", "adapter": "openai_chat_completions",
            "base_url": BASE_URL, "source": "local test fixture",
        } for model in ("first", "backup")}
        for target, value in (
            ("MODEL_PROVIDER_ALLOWLIST", ("test",)),
            ("MODEL_RESPONSE_CONTRACTS", contracts),
            ("MODEL_OUTPUT_BUDGETS", {}),
        ):
            self.enterContext(patch.object(config, target, value))
        self.enterContext(patch.object(config, "resolve_model_providers", return_value=self.providers))
        factory = self.enterContext(patch("src.ai.summarizer.OpenAI"))
        self.summarizer = Summarizer()
        self.client = factory.return_value
        self.call = self.client.chat.completions.create
        self.assertEqual(factory.call_args.kwargs["max_retries"], 0)

    def single(self, payload):
        self.call.return_value = payload
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = self.summarizer._call_llm(
                self.client, "first", "Course", "PRIVATE SOURCE TEXT",
                provider_name="test", base_url=BASE_URL,
            )
        return result, logs(out.getvalue())[-1]

    def modelscope_single(self, payload, *, model=MODELSCOPE_MODEL, base_url=MODELSCOPE_URL,
                          contracts=None):
        self.call.return_value = payload
        out = io.StringIO()
        with patch.object(config, "MODEL_RESPONSE_CONTRACTS",
                          PRODUCTION_CONTRACTS if contracts is None else contracts), \
                patch.object(config, "MODEL_OUTPUT_BUDGETS", PRODUCTION_BUDGETS), \
                contextlib.redirect_stdout(out):
            result = self.summarizer._call_llm(
                self.client, model, "Course", "PRIVATE SOURCE TEXT",
                provider_name="modelscope", base_url=base_url,
            )
        return result, logs(out.getvalue())[-1], out.getvalue()

    def test_verified_stop_with_nonempty_text_is_complete(self):
        result, record = self.single(response(text="  supplied proof  "))
        self.assertIsInstance(result, SummaryResult)
        self.assertTrue(result.is_complete)
        self.assertEqual(result.summary, "supplied proof")
        self.assertEqual(record["raw_finish_reason"], "stop")
        self.assertEqual(record["response_model"], "returned-model")
        self.assertEqual(record["budget_status"], "budget_unconfigured")
        self.assertEqual(set(self.call.call_args.kwargs), {"model", "messages", "timeout"})

    def test_length_is_not_a_success_even_with_nonempty_text(self):
        result, record = self.single(response("length", "Proof ends at x ="))
        self.assertFalse(result.is_complete)
        self.assertEqual(result.completion_status, "truncated")
        self.assertEqual(record["raw_finish_reason"], "length")

    def test_missing_none_and_unknown_reasons_remain_unknown(self):
        for reason in (MISSING, None, "", "future_reason", "STOP", {"unknown": 1}):
            with self.subTest(reason=reason):
                result, record = self.single(response(reason))
                self.assertFalse(result.is_complete)
                self.assertEqual(result.completion_status, "unknown")
                self.assertEqual(result.finish_reason, None if reason is MISSING else reason)
                if not isinstance(reason, dict):
                    self.assertEqual(record["raw_finish_reason"], None if reason is MISSING else reason)

    def test_abnormal_endings_and_refusals_are_rejected(self):
        for reason in ("content_filter", "tool_calls", "function_call", "aborted", "insufficient_system_resource"):
            with self.subTest(reason=reason):
                result, _ = self.single(response(reason))
                self.assertEqual(result.completion_status, "abnormal")
                self.assertFalse(result.is_complete)
        for field in ("refusal", "tool_calls", "function_call"):
            payload = response()
            payload["choices"][0]["message"][field] = "not a lecture completion"
            with self.subTest(field=field):
                result, _ = self.single(payload)
                self.assertFalse(result.is_complete)

    def test_empty_and_malformed_responses_cannot_succeed(self):
        for payload in (None, {}, {"choices": []}, {"choices": [None]},
                        response(text=""), response(text="   "), response(text=None)):
            with self.subTest(payload=payload):
                result, _ = self.single(payload)
                self.assertFalse(result.is_complete)

    def test_usage_missing_partial_and_zero_are_not_estimated(self):
        for usage in (None, {}, {"prompt_tokens": 0},
                      SimpleNamespace(prompt_tokens=0, completion_tokens=5)):
            with self.subTest(usage=usage):
                _, record = self.single(response(usage=usage))
                self.assertEqual(record["prompt_tokens"], 0 if usage else None)
                self.assertIsNone(record["total_tokens"])
                self.assertIsNone(record["reasoning_tokens"])

    def test_reasoning_is_a_detail_not_added_to_completion_or_total(self):
        usage = {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130,
                 "completion_tokens_details": {"reasoning_tokens": 20},
                 "prompt_tokens_details": {"cached_tokens": 25}}
        result, record = self.single(response(usage=usage))
        self.assertEqual(record["completion_tokens"], 30)
        self.assertEqual(record["reasoning_tokens"], 20)
        self.assertEqual(record["total_tokens"], 130)
        self.assertEqual(record["raw_usage"], usage)
        self.assertEqual(result.usage, usage)

    def test_top_level_reasoning_count_is_retained(self):
        _, record = self.single(response(usage={"completion_tokens": 8, "reasoning_tokens": 3}))
        self.assertEqual(record["reasoning_tokens"], 3)
        self.assertEqual(record["completion_tokens"], 8)
        self.assertIsNone(record["total_tokens"])

    def test_zero_nested_reasoning_is_not_replaced_or_added(self):
        _, record = self.single(response(usage={
            "completion_tokens": 8, "reasoning_tokens": 3,
            "completion_tokens_details": {"reasoning_tokens": 0},
        }))
        self.assertEqual(record["reasoning_tokens"], 0)
        self.assertEqual(record["completion_tokens"], 8)

    def test_usage_and_duration_are_recorded_for_rejected_candidates(self):
        with patch("src.ai.summarizer.time.monotonic", side_effect=[10.0, 10.125]):
            _, record = self.single(response("length", usage={"completion_tokens": 12}))
        self.assertEqual(record["completion_tokens"], 12)
        self.assertEqual(record["elapsed_seconds"], 0.125)

    def test_endpoint_override_cannot_borrow_a_verified_contract(self):
        self.call.return_value = response()
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.summarizer._call_llm(
                self.client, "first", "Course", "source", provider_name="test",
                base_url="https://unverified.invalid/v1",
            )
        self.assertEqual(result.contract_status, "needs_verification")
        self.assertFalse(result.is_complete)

    def test_unobserved_modelscope_and_gemini_contracts_are_not_assumed_verified(self):
        for provider, model in (("modelscope", "deepseek-ai/DeepSeek-V4-Pro"),
                                ("gemini", "gemini-2.5-flash")):
            with self.subTest(provider=provider), patch.object(config, "MODEL_RESPONSE_CONTRACTS", PRODUCTION_CONTRACTS):
                self.call.return_value = response()
                with contextlib.redirect_stdout(io.StringIO()):
                    result = self.summarizer._call_llm(
                        self.client, model, "Course", "source", provider_name=provider,
                        base_url=PRODUCTION_CONTRACTS[(provider, model)]["base_url"],
                    )
                self.assertEqual(result.contract_status, "needs_verification")
                self.assertFalse(result.is_complete)

    def test_observed_modelscope_stop_and_usage_do_not_verify_output_budget(self):
        actual = observed_modelscope_response()
        as_object = json.loads(json.dumps(actual), object_hook=lambda value: SimpleNamespace(**value))
        for payload in (actual, as_object):
            with self.subTest(shape=type(payload).__name__):
                result, record, output = self.modelscope_single(payload)
                self.assertTrue(result.is_complete)
                self.assertEqual(result.summary, "OK")
                self.assertEqual(result.finish_reason, "stop")
                self.assertEqual(result.response_model, MODELSCOPE_MODEL)
                self.assertEqual(record["model_contract_status"], "partially_verified")
                self.assertEqual(record["contract_scope"], "normal_finish_reason")
                self.assertEqual(record["normal_finish_reason_status"], "observed")
                self.assertEqual(record["truncation_finish_reason_status"], "needs_verification")
                self.assertEqual(record["budget_status"], "budget_unconfigured")
                self.assertEqual(record["budget_parameter_status"], "needs_verification")
                self.assertIsNone(record["requested_budget"])
                self.assertIsNone(record["budget_parameter"])
                self.assertEqual(record["raw_usage"], actual["usage"])
                self.assertEqual(result.usage, actual["usage"])
                self.assertEqual(record["prompt_tokens"], 14)
                self.assertEqual(record["completion_tokens"], 227)
                self.assertEqual(record["reasoning_tokens"], 221)
                self.assertEqual(record["text_tokens"], 227)
                self.assertEqual(record["total_tokens"], 241)
                self.assertEqual(set(self.call.call_args.kwargs), {"model", "messages", "timeout"})
                self.assertNotIn(PRIVATE_REASONING, result.summary + output)
                self.assertNotIn("reasoning_content", output)

    def test_observed_modelscope_rejects_unconfirmed_and_abnormal_endings(self):
        for reason, status in ((MISSING, "unknown"), (None, "unknown"), ("", "unknown"),
                               ("future_reason", "unknown"), ("STOP", "unknown"),
                               ("length", "truncated"), ("content_filter", "abnormal")):
            with self.subTest(reason=reason):
                payload = observed_modelscope_response()
                choice = payload["choices"][0]
                if reason is MISSING:
                    del choice["finish_reason"]
                else:
                    choice["finish_reason"] = reason
                result, record, output = self.modelscope_single(payload)
                self.assertFalse(result.is_complete)
                self.assertEqual(result.completion_status, status)
                self.assertEqual(result.finish_reason, None if reason is MISSING else reason)
                self.assertEqual(record["truncation_finish_reason_status"], "needs_verification")
                self.assertNotIn(PRIVATE_REASONING, output)

    def test_reasoning_content_cannot_replace_missing_modelscope_content(self):
        for text in (MISSING, None, "", "   "):
            with self.subTest(text=text):
                payload = observed_modelscope_response()
                message = payload["choices"][0]["message"]
                if text is MISSING:
                    del message["content"]
                else:
                    message["content"] = text
                result, _, output = self.modelscope_single(payload)
                self.assertFalse(result.is_complete)
                self.assertEqual(result.summary, "")
                self.assertEqual(result.completion_status, "abnormal")
                self.assertNotIn(PRIVATE_REASONING, output)
                self.assertNotIn("reasoning_content", output)

    def test_modelscope_observed_stop_is_scoped_to_exact_model_and_endpoint(self):
        for model, url in ((MODELSCOPE_MODEL, "https://unverified.invalid/v1/"),
                           ("Qwen/Qwen3.5-397B-A17B-other", MODELSCOPE_URL),
                           ("deepseek-ai/DeepSeek-V4-Pro", MODELSCOPE_URL)):
            with self.subTest(model=model, url=url):
                result, _, _ = self.modelscope_single(observed_modelscope_response(),
                                                     model=model, base_url=url)
                self.assertFalse(result.is_complete)
                self.assertEqual(result.contract_status, "needs_verification")

    def test_partial_model_contract_requires_explicit_observed_stop_rule(self):
        for field in ("normal_finish_reason_status", "normal_finish_reasons"):
            with self.subTest(field=field):
                contracts = copy.deepcopy(PRODUCTION_CONTRACTS)
                del contracts[("modelscope", MODELSCOPE_MODEL)][field]
                result, _, _ = self.modelscope_single(observed_modelscope_response(), contracts=contracts)
                self.assertFalse(result.is_complete)
                self.assertEqual(result.contract_status, "needs_verification")

    def test_usage_object_fallback_preserves_reasoning_and_text_counts_without_sum(self):
        class Counts:
            prompt_tokens = 14
            completion_tokens = 227
            total_tokens = 241
            completion_tokens_details = SimpleNamespace(reasoning_tokens=221, text_tokens=227)
        payload = observed_modelscope_response()
        payload["usage"] = Counts()
        result, record, _ = self.modelscope_single(payload)
        self.assertEqual(result.usage, observed_modelscope_response()["usage"])
        self.assertEqual(record["completion_tokens"], 227)
        self.assertEqual(record["text_tokens"], 227)
        self.assertEqual(record["reasoning_tokens"], 221)
        self.assertEqual(record["total_tokens"], 241)

    def test_reasoning_content_is_omitted_from_normal_and_error_logs(self):
        payload = observed_modelscope_response()
        payload["usage"]["vendor_debug"] = {"reasoning_content": PRIVATE_REASONING}
        _, _, output = self.modelscope_single(payload)
        self.assertNotIn(PRIVATE_REASONING, output)
        self.assertNotIn("reasoning_content", output)
        class LocalAPIError(Exception):
            status_code = 400
            body = payload
        self.call.side_effect = LocalAPIError(PRIVATE_REASONING)
        output = io.StringIO()
        with patch.object(config, "MODEL_RESPONSE_CONTRACTS", PRODUCTION_CONTRACTS), \
                patch.object(config, "MODEL_OUTPUT_BUDGETS", PRODUCTION_BUDGETS), \
                contextlib.redirect_stdout(output), self.assertRaises(SummaryAttemptError):
            self.summarizer._call_llm(
                self.client, MODELSCOPE_MODEL, "Course", "source",
                provider_name="modelscope", base_url=MODELSCOPE_URL,
            )
        self.assertNotIn(PRIVATE_REASONING, output.getvalue())
        self.assertNotIn("reasoning_content", output.getvalue())
        self.assertEqual(logs(output.getvalue())[0]["completion_tokens"], 227)

    def test_observed_modelscope_success_preserves_order_and_paid_provider_gate(self):
        self.summarizer.providers = [
            {"name": "modelscope", "base_url": MODELSCOPE_URL,
             "models": ["deepseek-ai/DeepSeek-V4-Pro", MODELSCOPE_MODEL]},
            {"name": "deepseek", "base_url": "https://api.deepseek.com/v1",
             "models": ["deepseek-v4-flash"]},
        ]
        paid = Mock()
        self.summarizer._clients = {"modelscope": self.client, "deepseek": paid}
        self.call.side_effect = [response(), observed_modelscope_response()]
        with patch.object(config, "MODEL_PROVIDER_ALLOWLIST", ("modelscope",)), \
                patch.object(config, "MODEL_RESPONSE_CONTRACTS", PRODUCTION_CONTRACTS), \
                patch.object(config, "MODEL_OUTPUT_BUDGETS", PRODUCTION_BUDGETS), \
                contextlib.redirect_stdout(io.StringIO()):
            result = self.summarizer.summarize("Course", "source")
        self.assertTrue(result.is_complete)
        self.assertEqual(result.summary, "OK")
        self.assertEqual([call.kwargs["model"] for call in self.call.call_args_list],
                         ["deepseek-ai/DeepSeek-V4-Pro", MODELSCOPE_MODEL])
        paid.chat.completions.create.assert_not_called()

    def test_deepseek_documented_shape_with_mock_response(self):
        with patch.object(config, "MODEL_RESPONSE_CONTRACTS", PRODUCTION_CONTRACTS):
            self.call.return_value = SimpleNamespace(
                model="deepseek-returned", choices=[SimpleNamespace(
                    finish_reason="stop", message=SimpleNamespace(content="source-supported proof"))],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=6, total_tokens=16,
                                      completion_tokens_details=SimpleNamespace(reasoning_tokens=2)),
            )
            with contextlib.redirect_stdout(io.StringIO()):
                result = self.summarizer._call_llm(
                    self.client, "deepseek-v4-flash", "Course", "source",
                    provider_name="deepseek", base_url="https://api.deepseek.com/v1",
                )
            self.assertTrue(result.is_complete)
            self.assertEqual(result.diagnostics["reasoning_tokens"], 2)

    def test_native_gemini_shape_is_diagnostic_only_at_chat_endpoint(self):
        usage = {"promptTokenCount": 10, "candidatesTokenCount": 6,
                 "thoughtsTokenCount": 2, "totalTokenCount": 18}
        payload = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "native text"}]}}],
                   "usageMetadata": usage}
        result, record = self.single(payload)
        self.assertFalse(result.is_complete)
        self.assertEqual(record["raw_finish_reason"], "STOP")
        self.assertEqual(record["raw_usage"], usage)
        self.assertEqual(record["reasoning_tokens"], 2)
        self.assertIsNone(record["completion_tokens"])
        self.assertEqual(record["candidate_tokens"], 6)
        self.assertEqual(record["total_tokens"], 18)

    def test_fallback_uses_each_allowed_model_once(self):
        self.providers[0]["models"] = ["first", "first", "backup", "backup"]
        self.call.side_effect = [response("length"), response()]
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.summarizer.summarize("Course", "source")
        self.assertTrue(result.is_complete)
        self.assertEqual(result.model_used, "test/backup")
        self.assertEqual([c.kwargs["model"] for c in self.call.call_args_list], ["first", "backup"])

    def test_all_models_fail_without_returning_a_partial_result(self):
        self.call.side_effect = [response("length"), response(None)]
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SummaryGenerationError) as caught:
            self.summarizer.summarize("Course", "source")
        self.assertEqual(self.call.call_count, 2)
        self.assertEqual(len(caught.exception.attempts), 2)
        self.assertEqual([x["classification"] for x in caught.exception.attempts], ["truncated", "unknown"])

    def test_empty_input_makes_no_requests(self):
        with self.assertRaises(SummaryGenerationError):
            self.summarizer.summarize("Course", " ")
        self.call.assert_not_called()

    def test_disabled_paid_provider_is_not_called_during_fallback(self):
        paid = Mock()
        self.summarizer.providers.append({"name": "deepseek", "models": ["deepseek-v4-flash"], "base_url": "https://api.deepseek.com/v1"})
        self.summarizer._clients["deepseek"] = paid
        self.call.side_effect = [response("length"), response("length")]
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SummaryGenerationError):
            self.summarizer.summarize("Course", "source")
        self.assertEqual(self.call.call_count, 2)
        paid.chat.completions.create.assert_not_called()

    def test_logs_do_not_dump_exception_source_or_answer(self):
        self.call.side_effect = RuntimeError("DO-NOT-LOG-PROVIDER-PAYLOAD")
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SummaryGenerationError):
            self.summarizer.summarize("Course", "PRIVATE SOURCE TEXT")
        self.assertNotIn("DO-NOT-LOG", out.getvalue())
        self.assertNotIn("PRIVATE SOURCE", out.getvalue())
        self.assertEqual(len(logs(out.getvalue())), 2)

    def test_usage_extensions_are_retained_in_memory_but_redacted_in_logs(self):
        usage = {"prompt_tokens": 12, "vendor_debug": {
            "request_content": "DO-NOT-LOG-METADATA", "api_key": "DO-NOT-LOG-CREDENTIAL",
        }}
        self.call.return_value = response(usage=usage)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = self.summarizer._call_llm(
                self.client, "first", "Course", "source", provider_name="test", base_url=BASE_URL,
            )
        self.assertEqual(result.usage, usage)
        self.assertEqual(result.diagnostics["raw_usage"], usage)
        self.assertNotIn("DO-NOT-LOG", out.getvalue())
        self.assertEqual(logs(out.getvalue())[0]["prompt_tokens"], 12)

    def test_error_response_usage_is_retained_without_dumping_private_fields(self):
        class LocalAPIError(Exception):
            status_code = 429
            body = response("length", usage={
                "prompt_tokens": 12, "completion_tokens": 2,
                "vendor_debug": "DO-NOT-LOG-ERROR-METADATA",
            })
        self.call.side_effect = LocalAPIError("DO-NOT-LOG-ERROR-MESSAGE")
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SummaryGenerationError) as caught:
            self.summarizer.summarize("Course", "PRIVATE SOURCE TEXT")
        self.assertEqual(len(caught.exception.attempts), 2)
        record = caught.exception.attempts[0]
        self.assertEqual(record["raw_finish_reason"], "length")
        self.assertEqual(record["prompt_tokens"], 12)
        self.assertEqual(record["completion_tokens"], 2)
        self.assertEqual(record["http_status"], 429)
        self.assertNotIn("DO-NOT-LOG", out.getvalue())
        self.assertNotIn("PRIVATE SOURCE", out.getvalue())

    def test_structured_unknown_reason_is_retained_without_dumping_text(self):
        reason = {"provider_debug": "DO-NOT-LOG-REASON"}
        self.call.return_value = response(reason=reason)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = self.summarizer._call_llm(
                self.client, "first", "Course", "source", provider_name="test", base_url=BASE_URL,
            )
        self.assertEqual(result.finish_reason, reason)
        self.assertFalse(result.is_complete)
        self.assertNotIn("DO-NOT-LOG", out.getvalue())

    def test_nonfinite_invalid_budget_does_not_break_fallback_logging(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                config.MODEL_OUTPUT_BUDGETS[("test", "first")] = {
                    "parameter": "max_tokens", "parameter_status": "verified",
                    "base_url": BASE_URL, "value": value,
                }
                self.call.reset_mock()
                self.call.return_value = response()
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    result = self.summarizer.summarize("Course", "source")
                self.assertEqual(result.model_used, "test/backup")
                self.assertEqual(self.call.call_count, 1)
                self.assertEqual(logs(out.getvalue())[0]["classification"], "abnormal")

    def test_budget_is_model_specific_and_only_one_length_field_is_sent(self):
        for parameter in ("max_tokens", "max_completion_tokens"):
            with self.subTest(parameter=parameter):
                config.MODEL_OUTPUT_BUDGETS[("test", "first")] = {
                    "parameter": parameter, "parameter_status": "verified",
                    "base_url": BASE_URL, "value": 512,
                }
                self.single(response())
                request = self.call.call_args.kwargs
                self.assertEqual(request[parameter], 512)
                self.assertEqual(set(request) & {"max_tokens", "max_completion_tokens"}, {parameter})
        self.assertNotIn(("test", "backup"), config.MODEL_OUTPUT_BUDGETS)

    def test_invalid_budget_does_not_trigger_an_unbounded_unbudgeted_retry(self):
        for value, parameter, status in ((0, "max_tokens", "verified"), (True, "max_tokens", "verified"),
                                         (512, "max_output_tokens", "verified"), (512, "max_tokens", "needs_verification")):
            with self.subTest(value=value, parameter=parameter, status=status):
                config.MODEL_OUTPUT_BUDGETS[("test", "first")] = {
                    "parameter": parameter, "parameter_status": status,
                    "base_url": BASE_URL, "value": value,
                }
                self.call.reset_mock()
                self.call.return_value = response()
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SummaryAttemptError):
                    self.summarizer._call_llm(self.client, "first", "Course", "source", provider_name="test", base_url=BASE_URL)
                self.call.assert_not_called()

    def test_provider_budget_error_falls_back_without_repeating_model(self):
        config.MODEL_OUTPUT_BUDGETS[("test", "first")] = {
            "parameter": "max_tokens", "parameter_status": "verified", "base_url": BASE_URL, "value": 512,
        }
        self.call.side_effect = [RuntimeError("unsupported budget"), response()]
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.summarizer.summarize("Course", "source")
        self.assertEqual(result.model_used, "test/backup")
        self.assertEqual([x.kwargs["model"] for x in self.call.call_args_list], ["first", "backup"])
        self.assertEqual(self.call.call_args_list[0].kwargs["max_tokens"], 512)


class ProviderConfigurationTests(unittest.TestCase):
    def test_production_budgets_are_all_unconfigured(self):
        self.assertTrue(config.MODEL_OUTPUT_BUDGETS)
        for policy in config.MODEL_OUTPUT_BUDGETS.values():
            self.assertIsNone(policy["value"])

    def test_keys_alone_do_not_enable_paid_fallback(self):
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key", "DEEPSEEK_API_KEY": "test-key", "GEMINI_API_KEY": "test-key"}, clear=True), \
                patch.object(config, "MODEL_PROVIDER_ALLOWLIST", ("modelscope",)):
            providers = config.resolve_model_providers()
        self.assertEqual([p["name"] for p in providers], ["modelscope"])
        self.assertEqual(providers[0]["models"], ["deepseek-ai/DeepSeek-V4-Pro", "Qwen/Qwen3.5-397B-A17B"])


if __name__ == "__main__":
    unittest.main()
