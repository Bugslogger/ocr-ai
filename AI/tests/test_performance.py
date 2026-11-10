"""Timing correlation and failure behavior without timing-sensitive assertions."""
import json
from unittest import TestCase
from AI.performance import performance_run, timed_stage, timed_pipeline


class PerformanceTests(TestCase):
    def test_nested_pipeline_stages_share_run_id_and_reset_between_calls(self):
        @timed_pipeline("nested")
        def nested():
            with timed_stage("extracting_pdf_text", pages=2):
                pass
        with self.assertLogs("AI.performance", level="INFO") as captured:
            with performance_run("invoice_job", run_id="test-job"):
                nested()
            nested()
        events = [json.loads(record.message.split("performance ", 1)[1]) for record in captured.records]
        self.assertTrue(all(event["run_id"] == "test-job" for event in events[:3]))
        self.assertTrue(all(event["run_id"] != "test-job" for event in events[3:]))
        self.assertTrue(all(event["elapsed_ms"] >= 0 for event in events))

    def test_failed_stage_records_failure_and_reraises(self):
        with self.assertLogs("AI.performance", level="INFO") as captured:
            with self.assertRaises(ValueError), performance_run("invoice_pipeline"):
                with timed_stage("extracting_json"):
                    raise ValueError("private invoice text")
        events = [json.loads(record.message.split("performance ", 1)[1]) for record in captured.records]
        self.assertEqual([event["status"] for event in events], ["failed", "failed"])
        self.assertEqual(events[0]["error_type"], "ValueError")
        self.assertNotIn("private invoice text", " ".join(captured.output))

    def test_job_that_handles_its_exception_can_report_failure(self):
        with self.assertLogs("AI.performance", level="INFO") as captured:
            with performance_run("invoice_job", run_id="failed-job") as metrics:
                metrics["stage_status"] = "failed"
        event = json.loads(captured.records[0].message.split("performance ", 1)[1])
        self.assertEqual(event["status"], "failed")
        self.assertEqual(event["run_id"], "failed-job")
