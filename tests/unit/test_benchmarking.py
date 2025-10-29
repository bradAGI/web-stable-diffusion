from __future__ import annotations

from web_stable_diffusion.runtime.benchmarking import summarise_benchmark


def test_summarise_benchmark_distils_core_metrics() -> None:
    benchmark = {
        "executor": "thread",
        "requested_executor": "auto",
        "max_workers": 4,
        "requested_max_workers": 8,
        "wall_clock_s": 0.42,
        "makespan_s": 0.42,
        "modalities_per_s": 9.5,
        "speedup_vs_serial": 2.1,
        "worker_utilisation": 0.77,
        "average_duration_s": 0.12,
        "completed": 4,
        "total": 4,
        "transport_breakdown": {"pinned_memory": 1, "host_transfer": 2, "zero_copy": 1},
        "resource_summary": {
            "memory_bytes_peak": 123456,
            "cpu_seconds_total": 1.68,
            "completed_modalities": 4,
        },
        "executor_state": {
            "queue_depth_initial": 4,
            "queue_depth_final": 0,
            "worker_launches": 4,
            "peak_in_flight": 3,
            "policy": "work-stealing",
        },
        "timeline": [
            {"type": "dispatch", "timestamp_s": 0.0},
            {"type": "complete", "timestamp_s": 0.1},
            {"type": "complete", "timestamp_s": 0.3},
        ],
        "tasks": {
            "audio": {"dispatch_seq": 0},
            "image": {"dispatch_seq": 1},
        },
    }

    summary = summarise_benchmark(benchmark)

    assert summary["executor"] == "thread"
    assert summary["requested_executor"] == "auto"
    assert summary["worker_count"] == 4
    assert summary["requested_worker_count"] == 8
    assert summary["modalities_completed"] == 4
    assert summary["modalities_total"] == 4
    assert summary["transport_breakdown"] == {
        "pinned_memory": 1,
        "host_transfer": 2,
        "zero_copy": 1,
    }
    assert summary["resource_summary"] == {
        "memory_bytes_peak": 123456.0,
        "cpu_seconds_total": 1.68,
        "completed_modalities": 4,
    }
    assert summary["executor_state"]["peak_in_flight"] == 3
    assert summary["timeline_events"] == 3
    assert 0.29 <= summary["timeline_span_s"] <= 0.3
    assert summary["tracked_modalities"] == 2
