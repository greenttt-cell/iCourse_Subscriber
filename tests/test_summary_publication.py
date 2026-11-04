"""Publication boundaries use temporary databases and simulated model results."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import main
from scripts.export_course import _query_course
from src.ai.summarizer import SummaryResult
from src.data.database import Database
from src.pipeline.lecture_runner import LectureRunner


def completed(text="Verified notes", model="test/model"):
    return SummaryResult(text, model, "complete", "stop", "verified")


def partial_source(db):
    # Build an initial partial fixture; the production API correctly refuses
    # to downgrade the complete transcript used by the common test setup.
    with db.conn:
        db.conn.execute("UPDATE lectures SET transcript = NULL WHERE sub_id = 'lecture'")
    assert db.update_transcript("lecture", "Original source", partial=True,
                                received_seconds=100, expected_seconds=300)


class SummaryPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.temp.name) / "lecture.db")
        self.db = Database(self.path)
        self.db.upsert_course("course", "Course", "Teacher")
        self.db.insert_lecture("lecture", "course", "Lecture", "2026-11-04")
        self.db.update_transcript("lecture", "Original source")
        self.model = Mock()
        self.runner = LectureRunner(Mock(), self.db, Mock(), Mock(), self.model, Mock())
        self.runner._ppt = Mock()
        self.lecture = {"sub_id": "lecture", "sub_title": "Lecture"}

    def tearDown(self):
        self.db.conn.close()
        self.temp.cleanup()

    def summarize(self):
        return self.runner._summarize("lecture", "Course", "Original source", None)

    def assert_no_published_output(self):
        row = self.db.get_lecture("lecture")
        self.assertIsNone(row["summary"])
        self.assertIsNone(row["summary_model"])
        self.assertIsNone(row["processed_at"])
        self.assertIsNone(row["emailed_at"])
        self.assertEqual(self.db.get_unsent_lectures(), [])
        self.assertIsNone(_query_course(self.db, "course"))
        emailer = Mock()
        main._send_email(emailer, self.db, Mock(), [])
        emailer.send.assert_not_called()

    def test_verified_stop_publishes_and_completes_in_one_transaction(self):
        self.db.update_error("lecture", "summarize", "Prior failure")
        self.model.summarize.return_value = completed()
        self.assertEqual(self.runner.run("course", "Course", self.lecture), "Verified notes")
        row = self.db.get_lecture("lecture")
        self.assertEqual(row["summary"], "Verified notes")
        self.assertEqual(row["summary_model"], "test/model")
        self.assertIsNotNone(row["processed_at"])
        self.assertIsNone(row["error_stage"])
        self.assertIsNone(row["error_msg"])
        self.assertIsNone(row["last_error_at"])
        self.assertEqual(row["error_count"], 0)
        self.assertEqual(len(self.db.get_unsent_lectures()), 1)
        self.assertEqual(_query_course(self.db, "course")[2][0]["summary"], "Verified notes")

    def test_invalid_receipts_and_legacy_tuple_never_reach_delivery_or_export(self):
        results = (
            SummaryResult("Cut off", "test/model", "truncated", "length", "verified"),
            SummaryResult("Unverified", "test/model", "complete", "stop", "needs_verification"),
            SummaryResult("", "test/model", "complete", "stop", "verified"),
            SummaryResult("Filtered", "test/model", "complete", "content_filter", "verified"),
            ("Legacy tuple", "test/model"),
        )
        for result in results:
            with self.subTest(result=result):
                self.model.summarize.return_value = result
                with self.assertRaisesRegex(ValueError, "unverified or incomplete"):
                    self.runner.run("course", "Course", self.lecture)
                self.assert_no_published_output()
                self.assertEqual(self.db.get_lecture("lecture")["error_stage"], "summarize")

    def test_all_backups_fail_without_publishing_or_marking_processed(self):
        self.model.summarize.side_effect = RuntimeError("All LLM models failed")
        with self.assertRaisesRegex(RuntimeError, "All LLM models failed"):
            self.runner.run("course", "Course", self.lecture)
        self.assert_no_published_output()
        row = self.db.get_lecture("lecture")
        self.assertEqual(row["error_count"], 1)
        self.assertEqual(row["error_stage"], "summarize")

    def test_existing_summary_is_returned_without_calling_failing_model(self):
        self.db.update_summary("lecture", "Existing notes", "old/model")
        self.db.mark_processed("lecture")
        self.db.mark_emailed("lecture")
        before = self.db.get_lecture("lecture")
        self.model.summarize.side_effect = RuntimeError("All LLM models failed")
        self.assertEqual(self.summarize(), "Existing notes")
        self.model.summarize.assert_not_called()
        self.assertEqual(self.db.get_lecture("lecture"), before)

    def test_partial_summary_is_reused_without_altering_source_error(self):
        partial_source(self.db)
        self.db.update_summary("lecture", "Existing partial notes", "old/model")
        self.db.update_error("lecture", "transcribe", "Partial audio")
        self.db.mark_emailed("lecture")
        before = self.db.get_lecture("lecture")
        self.assertEqual(self.summarize(), "Existing partial notes")
        self.model.summarize.assert_not_called()
        self.assertEqual(self.db.get_lecture("lecture"), before)

    def test_source_change_during_generation_discards_candidate(self):
        def generate(*args):
            self.db.update_transcript("lecture", "Replacement source")
            return completed()
        self.model.summarize.side_effect = generate
        with self.assertRaisesRegex(RuntimeError, "Source changed during"):
            self.runner.run("course", "Course", self.lecture)
        self.assert_no_published_output()
        self.assertEqual(self.db.get_lecture("lecture")["transcript"], "Replacement source")

    def test_stale_source_is_rejected_before_model_call(self):
        self.db.update_transcript("lecture", "Replacement source")
        with self.assertRaisesRegex(RuntimeError, "Transcript changed before"):
            self.summarize()
        self.model.summarize.assert_not_called()
        self.assert_no_published_output()

    def test_concurrent_formal_summary_wins_without_changing_its_state(self):
        other = Database(self.path)
        try:
            expected = {}
            def generate(*args):
                other.update_summary("lecture", "Concurrent notes", "winner/model")
                other.mark_processed("lecture")
                other.mark_emailed("lecture")
                expected.update(other.get_lecture("lecture"))
                return completed("Losing candidate")
            self.model.summarize.side_effect = generate
            self.assertEqual(self.runner.run("course", "Course", self.lecture), "Concurrent notes")
            self.assertEqual(self.db.get_lecture("lecture"), expected)
        finally:
            other.conn.close()

    def test_concurrent_formal_summary_survives_failed_model_attempt(self):
        other = Database(self.path)
        try:
            expected = {}
            def generate(*args):
                other.update_summary("lecture", "Concurrent notes", "winner/model")
                other.mark_processed("lecture")
                other.mark_emailed("lecture")
                expected.update(other.get_lecture("lecture"))
                raise RuntimeError("All LLM models failed")
            self.model.summarize.side_effect = generate
            self.assertEqual(self.runner.run("course", "Course", self.lecture), "Concurrent notes")
            self.assertEqual(self.db.get_lecture("lecture"), expected)
        finally:
            other.conn.close()

    def test_publication_between_error_read_and_write_is_preserved(self):
        other = Database(self.path)
        try:
            expected = {}
            record_error = self.db.record_summary_error_if_absent
            def publish_before_error(*args, **kwargs):
                other.update_summary("lecture", "Concurrent notes", "winner/model")
                other.mark_processed("lecture")
                other.mark_emailed("lecture")
                expected.update(other.get_lecture("lecture"))
                return record_error(*args, **kwargs)
            self.model.summarize.side_effect = RuntimeError("All LLM models failed")
            with patch.object(self.db, "record_summary_error_if_absent", side_effect=publish_before_error):
                self.assertEqual(self.runner.run("course", "Course", self.lecture), "Concurrent notes")
            self.assertEqual(self.db.get_lecture("lecture"), expected)
        finally:
            other.conn.close()

    def test_source_change_after_publication_is_not_marked_processed_again(self):
        publish = self.db.save_generated_summary_if_absent
        def publish_then_replace(*args, **kwargs):
            saved = publish(*args, **kwargs)
            self.db.update_transcript("lecture", "Replacement source")
            return saved
        self.model.summarize.return_value = completed()
        with patch.object(self.db, "save_generated_summary_if_absent", side_effect=publish_then_replace):
            self.runner.run("course", "Course", self.lecture)
        self.assert_no_published_output()

    def test_partial_publication_keeps_source_retry_and_error_state(self):
        partial_source(self.db)
        self.db.update_error("lecture", "transcribe", "Partial audio")
        before = self.db.get_lecture("lecture")
        self.model.summarize.return_value = completed()
        self.assertIn("音频不完整", self.summarize())
        row = self.db.get_lecture("lecture")
        self.assertIsNone(row["processed_at"])
        for field in ("error_stage", "error_msg", "error_count", "last_error_at", "emailed_at"):
            self.assertEqual(row[field], before[field], field)
        self.assertEqual(len(self.db.get_unsent_lectures()), 1)

    def test_partial_source_generation_failure_stays_retryable_without_publishing(self):
        partial_source(self.db)
        self.db.update_error("lecture", "transcribe", "Partial audio")
        before = self.db.get_lecture("lecture")
        self.model.summarize.side_effect = RuntimeError("All LLM models failed")
        with self.assertRaisesRegex(RuntimeError, "All LLM models failed"):
            self.summarize()
        self.assert_no_published_output()
        row = self.db.get_lecture("lecture")
        for field in ("transcript", "transcript_partial", "audio_received_seconds", "audio_expected_seconds"):
            self.assertEqual(row[field], before[field], field)
        self.assertTrue(row["transcript_partial"])
        self.assertEqual(row["error_stage"], "summarize")
        self.assertEqual(row["error_msg"], "All LLM models failed")
        self.assertEqual(row["error_count"], before["error_count"] + 1)
        self.assertIsNotNone(row["last_error_at"])
        self.assertEqual([item["sub_id"] for item in self.db.get_unprocessed_lectures("course")], ["lecture"])


class AtomicSummaryWriteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Database(str(Path(self.temp.name) / "lecture.db"))
        self.db.insert_lecture("lecture", "course", "Lecture", "2026-11-04")
        self.db.update_transcript("lecture", "Original source")

    def tearDown(self):
        self.db.conn.close()
        self.temp.cleanup()

    def publish(self, **kwargs):
        return self.db.save_generated_summary_if_absent(
            "lecture", "Candidate", "test/model",
            expected_transcript="Original source", expected_partial=False, **kwargs)

    def test_existing_summary_and_all_status_fields_are_untouched(self):
        self.db.update_summary("lecture", "Original notes", "original/model")
        self.db.mark_processed("lecture")
        self.db.mark_emailed("lecture")
        self.db.update_error("lecture", "process", "Existing status")
        before = self.db.get_lecture("lecture")
        self.assertFalse(self.publish())
        self.assertEqual(self.db.get_lecture("lecture"), before)

    def test_partial_state_change_prevents_stale_publication(self):
        partial_source(self.db)
        before = self.db.get_lecture("lecture")
        self.assertFalse(self.publish(expected_received_seconds=100, expected_expected_seconds=300))
        self.assertEqual(self.db.get_lecture("lecture"), before)

    def test_changed_partial_duration_prevents_stale_publication(self):
        partial_source(self.db)
        before = self.db.get_lecture("lecture")
        self.assertFalse(self.db.save_generated_summary_if_absent(
            "lecture", "Candidate", "test/model", expected_transcript="Original source",
            expected_partial=True, expected_received_seconds=50, expected_expected_seconds=300))
        self.assertEqual(self.db.get_lecture("lecture"), before)

    def test_two_connections_cannot_publish_over_one_another(self):
        other = Database(self.db.db_path)
        try:
            self.assertTrue(self.publish())
            before = self.db.get_lecture("lecture")
            self.assertFalse(other.save_generated_summary_if_absent(
                "lecture", "Second candidate", "other/model",
                expected_transcript="Original source", expected_partial=False))
            self.assertEqual(self.db.get_lecture("lecture"), before)
        finally:
            other.conn.close()
