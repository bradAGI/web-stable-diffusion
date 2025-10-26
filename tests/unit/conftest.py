import sys
import types
from typing import Any, Callable

import numpy as np
import pytest


class FakeNDArray:
    """A lightweight stand-in for tvm.nd.NDArray used in tests."""

    def __init__(self, data: Any, device: str = "cpu") -> None:
        self._array = np.array(data)
        self.device = device

    def copyto(self, device: str) -> "FakeNDArray":
        return FakeNDArray(self._array.copy(), device)

    def numpy(self) -> np.ndarray:
        return np.array(self._array)


class _FakeModule:
    def __init__(self) -> None:
        self._registry: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        self._registry[name] = fn

    def __getitem__(self, name: str) -> Callable[..., Any]:
        return self._registry[name]


class FakeVirtualMachine:
    def __init__(self, rt_mod: Any, device: str) -> None:
        self.rt_mod = rt_mod
        self.device = device
        self.module = _FakeModule()

    def __getitem__(self, name: str) -> Callable[..., Any]:
        return self.module[name]

    def invoke_stateful(self, func_name: str) -> None:  # pragma: no cover - trivial
        self.last_invoked = func_name

    def time_evaluator(self, *_args: Any, **_kwargs: Any) -> Callable[..., Any]:
        def evaluator(*_e_args: Any, **_e_kwargs: Any) -> str:
            return "0.0s"

        return evaluator

    def get_outputs(self, _func_name: str) -> list[Any]:  # pragma: no cover - trivial
        return []


@pytest.fixture
def fake_tvm(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    fake_nd = types.SimpleNamespace(
        NDArray=FakeNDArray,
        array=lambda data, device: FakeNDArray(data, device),
        empty=lambda shape, dtype, device: FakeNDArray(
            np.zeros(shape, dtype=dtype), device
        ),
    )

    fake_relax = types.SimpleNamespace(VirtualMachine=FakeVirtualMachine, vm=types.SimpleNamespace(VirtualMachine=FakeVirtualMachine))
    fake_rpc = types.SimpleNamespace(connect=lambda *args, **kwargs: (args, kwargs))
    fake_ir = types.SimpleNamespace(Array=list)
    fake_tir = types.SimpleNamespace(PrimFunc=type("PrimFunc", (), {}))

    fake = types.SimpleNamespace(
        nd=fake_nd,
        relax=fake_relax,
        rpc=fake_rpc,
        ir=fake_ir,
        tir=fake_tir,
        cpu=lambda: "cpu",
    )

    modules = {
        "tvm": fake,
        "tvm.nd": fake_nd,
        "tvm.relax": fake_relax,
        "tvm.rpc": fake_rpc,
        "tvm.ir": fake_ir,
        "tvm.tir": fake_tir,
    }

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    return fake
