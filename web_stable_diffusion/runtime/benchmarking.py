"""Utilities for distilling omni-modal execution benchmark metadata."""

from __future__ import annotations

from typing import Any, Dict, Mapping

__all__ = ["summarise_benchmark"]


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def summarise_benchmark(benchmark: Mapping[str, Any]) -> Dict[str, Any]:
    """Produce a JSON-serialisable summary of the execution benchmark.

    The omni-modal engine records an extensive benchmark structure that includes
    scheduling decisions, transport strategies, and resource statistics for each
    modality.  Persisting the entire payload in manifests can be noisy, so this
    helper distils the data into a stable subset of metrics that downstream
    dashboards and tests can rely on.
    """

    summary: Dict[str, Any] = {}

    float_fields = {
        "wall_clock_s": "wall_clock_s",
        "makespan_s": "makespan_s",
        "modalities_per_s": "modalities_per_s",
        "speedup_vs_serial": "speedup_vs_serial",
        "worker_utilisation": "worker_utilisation",
        "average_duration_s": "average_duration_s",
    }
    for source, target in float_fields.items():
        value = _as_float(benchmark.get(source))
        if value is not None:
            summary[target] = value

    completed = _as_int(benchmark.get("completed"))
    if completed is not None:
        summary["modalities_completed"] = completed
    total = _as_int(benchmark.get("total"))
    if total is not None:
        summary["modalities_total"] = total

    worker_count = _as_int(benchmark.get("max_workers"))
    if worker_count is not None:
        summary["worker_count"] = worker_count
    requested_workers = _as_int(benchmark.get("requested_max_workers"))
    if requested_workers is not None:
        summary["requested_worker_count"] = requested_workers

    executor = benchmark.get("executor")
    if isinstance(executor, str):
        summary["executor"] = executor
    requested_executor = benchmark.get("requested_executor")
    if isinstance(requested_executor, str):
        summary["requested_executor"] = requested_executor

    deadline = _as_float(benchmark.get("deadline_s"))
    if deadline is not None:
        summary["deadline_s"] = deadline

    for key in ("timed_out", "cancelled", "fallback_to_threads", "supports_cuda_ipc"):
        value = benchmark.get(key)
        if isinstance(value, bool):
            summary[key] = value

    device = benchmark.get("device")
    if isinstance(device, Mapping):
        device_summary: Dict[str, Any] = {}
        backend = device.get("backend")
        if isinstance(backend, str):
            device_summary["backend"] = backend
        device_id = device.get("device")
        if isinstance(device_id, str):
            device_summary["device"] = device_id
        if device_summary:
            summary["device"] = device_summary

    transport = benchmark.get("transport_breakdown")
    if isinstance(transport, Mapping):
        summary["transport_breakdown"] = {
            "pinned_memory": _as_int(transport.get("pinned_memory")) or 0,
            "host_transfer": _as_int(transport.get("host_transfer")) or 0,
            "zero_copy": _as_int(transport.get("zero_copy")) or 0,
        }

    resource_summary = benchmark.get("resource_summary")
    if isinstance(resource_summary, Mapping):
        resources: Dict[str, Any] = {}
        memory_peak = _as_float(resource_summary.get("memory_bytes_peak"))
        if memory_peak is not None:
            resources["memory_bytes_peak"] = memory_peak
        cpu_total = _as_float(resource_summary.get("cpu_seconds_total"))
        if cpu_total is not None:
            resources["cpu_seconds_total"] = cpu_total
        completed_modalities = _as_int(resource_summary.get("completed_modalities"))
        if completed_modalities is not None:
            resources["completed_modalities"] = completed_modalities
        if resources:
            summary["resource_summary"] = resources

    executor_state = benchmark.get("executor_state")
    if isinstance(executor_state, Mapping):
        state: Dict[str, Any] = {}
        for key in ("queue_depth_initial", "queue_depth_final", "worker_launches", "peak_in_flight"):
            value = _as_int(executor_state.get(key))
            if value is not None:
                state[key] = value
        policy = executor_state.get("policy")
        if isinstance(policy, str):
            state["policy"] = policy
        if state:
            summary["executor_state"] = state

    timeline = benchmark.get("timeline")
    if isinstance(timeline, list):
        summary["timeline_events"] = len(timeline)
        if timeline:
            timestamps = [
                _as_float(event.get("timestamp_s"))
                for event in timeline
                if isinstance(event, Mapping)
            ]
            if timestamps:
                summary["timeline_span_s"] = max(timestamps) - min(timestamps)

    tasks = benchmark.get("tasks")
    if isinstance(tasks, Mapping):
        summary["tracked_modalities"] = len(tasks)

    return summary
