import sys
import importlib.util
from pathlib import Path

import pytest


def load_rpc_testing_module() -> object:
    module_name = "rpc_testing_under_test"
    module_path = Path(__file__).resolve().parents[2] / "web_stable_diffusion" / "rpc_testing.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_connect_to_proxy_reads_wasm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_tvm):
    wasm_file = tmp_path / "mock.wasm"
    wasm_file.write_bytes(b"wasm")

    calls = []

    def fake_connect(host, port, key=None, session_constructor_args=None):
        calls.append((host, port, key, session_constructor_args))
        return "remote-session"

    fake_tvm.rpc.connect = fake_connect
    monkeypatch.setitem(sys.modules, "tvm.rpc", fake_tvm.rpc)

    module = load_rpc_testing_module()

    remote = module.connect_to_proxy(str(wasm_file))

    assert remote == "remote-session"
    assert calls[0][0] == "127.0.0.1"
    assert calls[0][1] == 9090
    assert calls[0][2] == "wasm"
    assert calls[0][3][0] == "rpc.WasmSession"
    assert calls[0][3][1] == wasm_file.read_bytes()


def test_webgpu_debug_session_initializes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_tvm):
    wasm_file = tmp_path / "mock.wasm"
    wasm_file.write_bytes(b"wasm")

    class DummyRemote:
        def __init__(self):
            self.connected = True

        def system_lib(self):
            return "rt_mod"

        def webgpu(self):
            return "webgpu-device"

    dummy_remote = DummyRemote()

    module = load_rpc_testing_module()

    monkeypatch.setattr(module, "connect_to_proxy", lambda _: dummy_remote)

    session = module.WebGPUDebugSession(str(wasm_file))

    assert session.remote is dummy_remote
    assert session.device == "webgpu-device"
    assert session.vm.rt_mod == "rt_mod"
