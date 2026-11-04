"""Regression checks for mathematical content preservation; no model calls."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from src.ai.ppt_dedup import clean_ppt_text
from src.ai.summarizer import SYSTEM_PROMPT, Summarizer
from src.ai.transcriber import _postprocess_segment


class TranscriptMathTests(unittest.TestCase):
    def test_variables_and_logical_conditions_survive(self):
        samples = (
            "Let A be a matrix.",
            "If and only if",
            "only if",
            "This holds if and only if the domain is compact.",
            "取 A 为矩阵",
            "A = 1, B = 2.",
            "x < y > z",
            "<x, y>",
            "<|x|>",
            "取 A 为矩阵，Let x = Ay，仅当 x ≠ 0 时成立。",
        )
        for text in samples:
            with self.subTest(text=text):
                self.assertEqual(_postprocess_segment(text), text)

    def test_standalone_variables_numbers_and_operators_survive(self):
        for text in ("A", "B", "x", "y", "0", "1", "∑", "α", "∞"):
            with self.subTest(text=text):
                self.assertEqual(_postprocess_segment(text), text)

    def test_latex_and_mixed_language_are_unchanged(self):
        samples = (
            r"Let $A \in \mathbb{R}^{n\times n}$ be a matrix.",
            r"仅当 $x > 0$ 时，$\log x$ 有定义；if and only if $A = A^T$。",
            r"\(\text{a matrix}\) 与 \[x < y,\quad y > z\]",
            "$$\n\\begin{aligned}\nA &= B \\\\\nx &= 0\n\\end{aligned}\n$$",
        )
        for text in samples:
            with self.subTest(text=text):
                self.assertEqual(_postprocess_segment(text), text)

    def test_only_known_asr_tags_are_removed(self):
        text = "<sil><|zh|><|HAPPY|><|Speech|>取 A 为矩阵"
        self.assertEqual(_postprocess_segment(text), "取 A 为矩阵")
        self.assertEqual(_postprocess_segment("<custom> x < y"), "<custom> x < y")

    def test_unambiguous_standalone_fillers_are_still_removed(self):
        for text in ("Yeah.", "嗯，呃。"):
            with self.subTest(text=text):
                self.assertEqual(_postprocess_segment(text), "")


class OCRMathTests(unittest.TestCase):
    def test_single_character_lines_survive(self):
        text = "定义\nA\nB\nx\ny\n0\n1\n∑\n□\n三\n结论"
        self.assertEqual(clean_ppt_text(text), text)

    def test_ambiguous_math_notation_and_headings_survive(self):
        text = "公式\n符号\n排列\n选择\n比较\n问题\n问题70\n输出\n转换\nA.\nAB\nEM\nPN\n62%\n1/2\nUabexx²²²"
        self.assertEqual(clean_ppt_text(text), text)

    def test_latex_and_mixed_language_survive(self):
        samples = (
            r"\frac{1}{2} \leq x < y,\quad x \in \mathbb{R}",
            r"取 $A$ 为矩阵，if and only if $A^T=A$，则 $A$ 对称。",
            "$$\n\\begin{aligned}\nA &= B \\\\\nx &= 0\n\\end{aligned}\n$$",
            "α ∈ ℝ\nx ≥ 0\ny ≠ 1\nf(x) → ∞",
        )
        for text in samples:
            with self.subTest(text=text):
                self.assertEqual(clean_ppt_text(text), text)

    def test_explicit_ui_noise_is_removed_around_math(self):
        text = "文件\n幻灯片放映\n单击此处添加标题\n第2页，共5页\nx\nf(x)=0"
        self.assertEqual(clean_ppt_text(text), "x\nf(x)=0")

    def test_multiline_latex_protects_text_rows_and_internal_spacing(self):
        blocks = (
            "$$\n\\text{\n替换\n}\n$$",
            "\\[\n\\text{\n接受\n}\n\\]",
            "\\begin{array}{c}\n  Usage  \n\n\\end{array}",
            "\\operatorname{\nUsage\n}",
            "$\n\\text{\n文件\n}\n$",
        )
        for block in blocks:
            with self.subTest(block=block):
                self.assertEqual(clean_ppt_text("文件\n" + block + "\n幻灯片放映"), block)
                self.assertEqual(clean_ppt_text(clean_ppt_text(block)), block)

    def test_incomplete_latex_is_retained_without_guessing(self):
        for block in ("$$\n替换", "\\[\n接受", "\\operatorname{\nUsage"):
            with self.subTest(block=block):
                self.assertEqual(clean_ppt_text(block), block)

    def test_empty_whitespace_and_repeated_cleaning(self):
        self.assertEqual(clean_ppt_text(""), "")
        self.assertEqual(clean_ppt_text(" \n\t"), "")
        text = "  x  \n文件\n∑\n1/2"
        cleaned = clean_ppt_text(text)
        self.assertEqual(cleaned, "x\n∑\n1/2")
        self.assertEqual(clean_ppt_text(cleaned), cleaned)


class MathPromptTests(unittest.TestCase):
    def test_requests_do_not_derive_a_word_target_from_input_length(self):
        summarizer = Summarizer.__new__(Summarizer)
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="本地模拟讲义"))],
            usage=None,
        )
        instruction = None
        for content in ("取 A 为矩阵", "取 A 为矩阵\n" * 1000):
            with self.subTest(input_chars=len(content)):
                self.assertEqual(
                    summarizer._call_llm(client, "mock-model", "数学课", content),
                    "本地模拟讲义",
                )
                request = client.chat.completions.create.call_args.kwargs
                self.assertEqual(set(request), {"model", "messages", "timeout"})
                self.assertEqual(request["messages"][0]["content"], SYSTEM_PROMPT)
                prefix, supplied = request["messages"][1]["content"].split("\n\n", 1)
                self.assertEqual(supplied, content)
                self.assertNotIn("字符数", prefix)
                if instruction is None:
                    instruction = prefix
                self.assertEqual(prefix, instruction)


if __name__ == "__main__":
    unittest.main()
