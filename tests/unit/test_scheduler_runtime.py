import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from tests.unit.conftest import FakeNDArray


def load_scheduler_runtime_module():
    module_name = "scheduler_runtime_under_test"
    module_path = (
        Path(__file__).resolve().parents[2]
        / "web_stable_diffusion"
        / "runtime"
        / "scheduler_runtime.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DummyVM:
    def __init__(self, return_value: float) -> None:
        self.return_value = return_value
        self.calls: list[tuple[str, tuple]] = []
        self._registry: dict[str, callable] = {}

    def register(self, name: str, fn) -> None:
        def wrapper(*args):
            self.calls.append((name, args))
            return fn(*args)

        self._registry[name] = wrapper

    def __getitem__(self, name: str):
        if name not in self._registry:
            def _default(*args):
                self.calls.append((name, args))
                return FakeNDArray(
                    np.full((1, 4, 2, 2), self.return_value, dtype=np.float32),
                    device="cpu",
                )

            self._registry[name] = _default
        return self._registry[name]


def _write_scheduler_consts(path: Path, filename: str, payload: dict) -> None:
    target = path / filename
    target.write_text(json.dumps(payload))


def test_pndm_scheduler_step_tracks_state(tmp_path: Path, fake_tvm) -> None:
    payload = {
        "timesteps": [[0], [1], [2], [3]],
        "sample_coeff": [[1.0]] * 4,
        "alpha_diff": [[0.9]] * 4,
        "model_output_denom_coeff": [[0.5]] * 4,
    }
    _write_scheduler_consts(tmp_path, "scheduler_pndm_consts.json", payload)

    module = load_scheduler_runtime_module()

    scheduler = module.PNDMScheduler(str(tmp_path), device="cpu")
    vm = DummyVM(return_value=0.25)

    sample = FakeNDArray(np.zeros((1, 4, 2, 2)), device="cpu")
    model_output = FakeNDArray(np.ones((1, 4, 2, 2)), device="cpu")

    first = scheduler.step(vm, model_output, sample, counter=0)
    second = scheduler.step(vm, model_output, sample, counter=1)

    assert isinstance(first, FakeNDArray)
    assert isinstance(second, FakeNDArray)
    assert len(vm.calls) == 2
    called_names = [name for name, _ in vm.calls]
    assert called_names[0] == "pndm_scheduler_step_0"
    assert called_names[1] == "pndm_scheduler_step_1"


def test_dpm_scheduler_converts_and_caches_output(tmp_path: Path, fake_tvm) -> None:
    payload = {
        "timesteps": [[0], [1], [2]],
        "alpha": [[0.9]] * 3,
        "sigma": [[0.1]] * 3,
        "c0": [[1.0]] * 3,
        "c1": [[0.5]] * 3,
        "c2": [[0.2]] * 3,
    }
    _write_scheduler_consts(
        tmp_path, "scheduler_dpm_solver_multistep_consts.json", payload
    )

    module = load_scheduler_runtime_module()

    scheduler = module.DPMSolverMultistepScheduler(str(tmp_path), device="cpu")
    vm = DummyVM(return_value=0.75)

    def convert(sample, model_output, alpha, sigma):
        return FakeNDArray(np.full((1, 4, 2, 2), 0.5, dtype=np.float32), "cpu")

    def step(sample, model_output, last_output, c0, c1, c2):
        return FakeNDArray(np.full((1, 4, 2, 2), 0.75, dtype=np.float32), "cpu")

    vm.register("dpm_solver_multistep_scheduler_convert_model_output", convert)
    vm.register("dpm_solver_multistep_scheduler_step", step)

    sample = FakeNDArray(np.zeros((1, 4, 2, 2)), device="cpu")
    model_output = FakeNDArray(np.ones((1, 4, 2, 2)), device="cpu")

    result = scheduler.step(vm, model_output, sample, counter=0)

    assert isinstance(result, FakeNDArray)
    assert vm.calls[0][0] == "dpm_solver_multistep_scheduler_convert_model_output"
    assert vm.calls[1][0] == "dpm_solver_multistep_scheduler_step"
    assert np.allclose(result.numpy(), np.full((1, 4, 2, 2), 0.75, dtype=np.float32))
