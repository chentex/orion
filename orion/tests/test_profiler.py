"""Tests for orion.profiler module."""

import json
import threading

from orion.profiler import NoOpProfiler, QueryProfiler, QueryRecord


class TestQueryRecord:  # pylint: disable=too-few-public-methods
    """Tests for the QueryRecord dataclass."""

    def test_query_record_to_dict(self):
        """Verify all fields round-trip through to_dict."""
        record = QueryRecord(
            sequence=1,
            query_type="metadata",
            metric_name="cpuUsage",
            agg_type="avg",
            uuid_count=5,
            index="ripsaw-kube-burner-*",
            wall_time_ms=123.45,
            connect_time_ms=10.0,
            tls_time_ms=5.0,
            ttfb_ms=80.0,
            transfer_time_ms=28.45,
            response_bytes=2048,
            query_body_bytes=512,
            request_path="/_search",
            http_status=200,
            timestamp="2026-04-29T12:00:00+00:00",
            is_pooled_connection=False,
        )
        result = record.to_dict()
        assert isinstance(result, dict)
        assert result["sequence"] == 1
        assert result["query_type"] == "metadata"
        assert result["metric_name"] == "cpuUsage"
        assert result["agg_type"] == "avg"
        assert result["uuid_count"] == 5
        assert result["index"] == "ripsaw-kube-burner-*"
        assert result["wall_time_ms"] == 123.45
        assert result["connect_time_ms"] == 10.0
        assert result["tls_time_ms"] == 5.0
        assert result["ttfb_ms"] == 80.0
        assert result["transfer_time_ms"] == 28.45
        assert result["response_bytes"] == 2048
        assert result["query_body_bytes"] == 512
        assert result["request_path"] == "/_search"
        assert result["http_status"] == 200
        assert result["timestamp"] == "2026-04-29T12:00:00+00:00"
        assert result["is_pooled_connection"] is False


class TestQueryProfiler:
    """Tests for the QueryProfiler class."""

    def _make_timing(self, **overrides):
        """Helper to build a timing kwargs dict with defaults."""
        defaults = {
            "wall_time_ms": 100.0,
            "connect_time_ms": 10.0,
            "tls_time_ms": 5.0,
            "ttfb_ms": 70.0,
            "transfer_time_ms": 15.0,
            "response_bytes": 1024,
            "query_body_bytes": 256,
            "request_path": "/_search",
            "http_status": 200,
            "is_pooled_connection": False,
        }
        defaults.update(overrides)
        return defaults

    def test_query_profiler_set_context_and_record(self):
        """set_context then record: verify fields are populated correctly."""
        profiler = QueryProfiler()
        profiler.set_context(
            query_type="metric",
            metric_name="cpuUsage",
            agg_type="avg",
            uuid_count=3,
            index="ripsaw-*",
        )
        profiler.record(**self._make_timing())

        records = profiler.records
        assert len(records) == 1
        rec = records[0]
        assert rec.query_type == "metric"
        assert rec.metric_name == "cpuUsage"
        assert rec.agg_type == "avg"
        assert rec.uuid_count == 3
        assert rec.index == "ripsaw-*"
        assert rec.wall_time_ms == 100.0
        assert rec.http_status == 200
        assert rec.timestamp  # non-empty

    def test_query_profiler_auto_increments_sequence(self):
        """Three records should have sequence 1, 2, 3."""
        profiler = QueryProfiler()
        for _ in range(3):
            profiler.set_context(query_type="metadata", index="idx")
            profiler.record(**self._make_timing())

        seqs = [r.sequence for r in profiler.records]
        assert seqs == [1, 2, 3]

    def test_query_profiler_to_json_structure(self):
        """Two records: verify full JSON structure including summary."""
        profiler = QueryProfiler()

        profiler.set_context(query_type="metadata", index="perf_scale_ci*")
        profiler.record(**self._make_timing(wall_time_ms=50.0, is_pooled_connection=True))

        profiler.set_context(query_type="metric", metric_name="cpu", agg_type="avg", index="ripsaw-*")
        profiler.record(**self._make_timing(wall_time_ms=150.0, is_pooled_connection=False))

        raw = profiler.to_json()
        data = json.loads(raw)

        # Top-level fields
        assert data["profile_version"] == "1.0"
        assert "run_timestamp" in data
        assert data["total_queries"] == 2
        assert data["total_es_time_ms"] == 200.0

        # Queries list
        assert len(data["queries"]) == 2

        # Summary - by_query_type
        summary = data["summary"]
        by_type = summary["by_query_type"]
        assert "metadata" in by_type
        assert by_type["metadata"]["count"] == 1
        assert by_type["metadata"]["total_ms"] == 50.0
        assert by_type["metadata"]["avg_ms"] == 50.0
        assert "metric" in by_type
        assert by_type["metric"]["count"] == 1

        # Summary - by_timing_phase
        phases = summary["by_timing_phase"]
        for phase in ("connect", "tls", "ttfb", "transfer"):
            assert phase in phases

        # Summary - connection_reuse_rate
        reuse = summary["connection_reuse_rate"]
        assert reuse["pooled"] == 1
        assert reuse["total"] == 2

    def test_query_profiler_thread_safety(self):
        """4 threads x 50 records = 200 total, no errors."""
        profiler = QueryProfiler()
        errors = []

        def worker():
            try:
                for _ in range(50):
                    profiler.set_context(query_type="metric", index="idx")
                    profiler.record(
                        wall_time_ms=1.0,
                        connect_time_ms=0.1,
                        tls_time_ms=0.1,
                        ttfb_ms=0.5,
                        transfer_time_ms=0.3,
                        response_bytes=100,
                        query_body_bytes=50,
                        request_path="/_search",
                        http_status=200,
                        is_pooled_connection=False,
                    )
            except Exception as exc:  # pylint: disable=broad-except
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(profiler.records) == 200

    def test_query_profiler_record_without_context_uses_defaults(self):
        """record() without prior set_context() uses query_type='unknown'."""
        profiler = QueryProfiler()
        profiler.record(**self._make_timing())

        rec = profiler.records[0]
        assert rec.query_type == "unknown"
        assert rec.metric_name is None


class TestNoOpProfiler:  # pylint: disable=too-few-public-methods
    """Tests for the NoOpProfiler class."""

    def test_noop_profiler_is_silent(self):
        """set_context + record + to_json returns '{}'."""
        profiler = NoOpProfiler()
        profiler.set_context(query_type="metric", index="idx")
        profiler.record(
            wall_time_ms=100.0,
            connect_time_ms=10.0,
            tls_time_ms=5.0,
            ttfb_ms=70.0,
            transfer_time_ms=15.0,
            response_bytes=1024,
            query_body_bytes=256,
            request_path="/_search",
            http_status=200,
            is_pooled_connection=False,
        )
        assert profiler.to_json() == "{}"
