"""Profiler for Elasticsearch / OpenSearch query timing instrumentation."""

import json
import threading
import time  # pylint: disable=unused-import  # needed by future instrumentation hooks
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass
class QueryRecord:  # pylint: disable=too-many-instance-attributes
    """Single recorded ES query with timing and context."""

    sequence: int
    query_type: str
    metric_name: Optional[str]
    agg_type: Optional[str]
    uuid_count: int
    index: str
    wall_time_ms: float
    connect_time_ms: float
    tls_time_ms: float
    ttfb_ms: float
    transfer_time_ms: float
    response_bytes: int
    query_body_bytes: int
    request_path: str
    http_status: int
    timestamp: str
    is_pooled_connection: bool

    def to_dict(self):
        """Return a plain dict of all fields."""
        return asdict(self)


class QueryProfiler:
    """Thread-safe profiler that collects ES query timing records."""

    def __init__(self):
        self._lock = threading.Lock()
        self._records = []
        self._sequence = 0
        self._pending_context = {}
        self._run_timestamp = datetime.now(timezone.utc).isoformat()

    @property
    def records(self):
        """Return a copy of the recorded queries."""
        with self._lock:
            return list(self._records)

    def set_context(self, **kwargs):
        """Store pending context for the next record() call."""
        with self._lock:
            self._pending_context = dict(kwargs)

    def record(self, **timing):
        """Create a QueryRecord from pending context + timing, append it."""
        with self._lock:
            self._sequence += 1
            ctx = self._pending_context
            rec = QueryRecord(
                sequence=self._sequence,
                query_type=ctx.get("query_type", "unknown"),
                metric_name=ctx.get("metric_name"),
                agg_type=ctx.get("agg_type"),
                uuid_count=ctx.get("uuid_count", 0),
                index=ctx.get("index", ""),
                wall_time_ms=timing["wall_time_ms"],
                connect_time_ms=timing["connect_time_ms"],
                tls_time_ms=timing["tls_time_ms"],
                ttfb_ms=timing["ttfb_ms"],
                transfer_time_ms=timing["transfer_time_ms"],
                response_bytes=timing["response_bytes"],
                query_body_bytes=timing["query_body_bytes"],
                request_path=timing["request_path"],
                http_status=timing["http_status"],
                timestamp=datetime.now(timezone.utc).isoformat(),
                is_pooled_connection=timing["is_pooled_connection"],
            )
            self._records.append(rec)
            self._pending_context = {}

    def to_json(self):
        """Generate a full JSON profiling report."""
        with self._lock:
            records = list(self._records)

        total_es_time = round(sum(r.wall_time_ms for r in records), 2)

        # by_query_type
        type_buckets = defaultdict(list)
        for rec in records:
            type_buckets[rec.query_type].append(rec.wall_time_ms)

        by_query_type = {}
        for qtype, times in type_buckets.items():
            by_query_type[qtype] = {
                "count": len(times),
                "total_ms": round(sum(times), 2),
                "avg_ms": round(sum(times) / len(times), 2),
            }

        # by_timing_phase
        by_timing_phase = {
            "connect": round(sum(r.connect_time_ms for r in records), 2),
            "tls": round(sum(r.tls_time_ms for r in records), 2),
            "ttfb": round(sum(r.ttfb_ms for r in records), 2),
            "transfer": round(sum(r.transfer_time_ms for r in records), 2),
        }

        # connection_reuse_rate
        pooled = sum(1 for r in records if r.is_pooled_connection)
        connection_reuse_rate = {
            "pooled": pooled,
            "total": len(records),
        }

        report = {
            "profile_version": "1.0",
            "run_timestamp": self._run_timestamp,
            "total_es_time_ms": total_es_time,
            "total_queries": len(records),
            "queries": [r.to_dict() for r in records],
            "summary": {
                "by_query_type": by_query_type,
                "by_timing_phase": by_timing_phase,
                "connection_reuse_rate": connection_reuse_rate,
            },
        }
        return json.dumps(report, indent=2)


class NoOpProfiler:
    """Drop-in replacement that does nothing; used when profiling is disabled."""

    def set_context(self, **kwargs):
        """No-op."""

    def record(self, **timing):
        """No-op."""

    def to_json(self):
        """Return empty JSON object."""
        return "{}"
