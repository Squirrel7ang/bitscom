import pytest

import bitscom.lowbit_backend as lb


def test_register_lowbit_backend_success(monkeypatch):
    recorded = {}

    def fake_register_backend(name, func, devices=None):
        recorded["name"] = name
        recorded["func"] = func
        recorded["devices"] = devices

    monkeypatch.setattr(lb, "_REGISTERED", False)
    monkeypatch.setattr(lb, "_HAS_EXTENSION", True)
    monkeypatch.setattr(lb, "_BACKEND_BITWIDTH", 4)
    monkeypatch.setattr(lb, "_BACKEND_ERROR_FEEDBACK", False)
    monkeypatch.setattr(lb, "_BACKEND_BLOCK_SIZE", 256)
    monkeypatch.setattr(lb, "_BACKEND_STAGE2_ERROR_FEEDBACK", False)
    monkeypatch.setattr(lb.dist.Backend, "register_backend", fake_register_backend)

    lb.register_lowbit_backend(bitwidth=2, error_feedback=True)

    assert recorded["name"] == "lowbit"
    assert callable(recorded["func"])
    assert recorded["devices"] == ["cpu", "cuda"]
    assert lb._BACKEND_BITWIDTH == 2
    assert lb._BACKEND_ERROR_FEEDBACK is True
    assert lb._BACKEND_BLOCK_SIZE == 256
    assert lb._BACKEND_STAGE2_ERROR_FEEDBACK is False
    assert lb._REGISTERED is True


def test_register_lowbit_backend_idempotent(monkeypatch):
    monkeypatch.setattr(lb, "_REGISTERED", True)
    monkeypatch.setattr(lb, "_BACKEND_BITWIDTH", 4)
    monkeypatch.setattr(lb, "_BACKEND_ERROR_FEEDBACK", False)
    monkeypatch.setattr(lb, "_BACKEND_ERROR_FEEDBACK_MODE", "none")
    monkeypatch.setattr(lb, "_BACKEND_BLOCK_SIZE", 256)
    monkeypatch.setattr(lb, "_BACKEND_STAGE2_ERROR_FEEDBACK", False)

    called = {"count": 0}

    def fake_register_backend(name, func, devices=None):
        called["count"] += 1

    monkeypatch.setattr(lb.dist.Backend, "register_backend", fake_register_backend)

    lb.register_lowbit_backend(bitwidth=4, error_feedback=False)
    assert called["count"] == 0


def test_register_lowbit_backend_rejects_conflicting_options(monkeypatch):
    monkeypatch.setattr(lb, "_REGISTERED", True)
    monkeypatch.setattr(lb, "_BACKEND_BITWIDTH", 4)
    monkeypatch.setattr(lb, "_BACKEND_ERROR_FEEDBACK", False)
    monkeypatch.setattr(lb, "_BACKEND_ERROR_FEEDBACK_MODE", "none")
    monkeypatch.setattr(lb, "_BACKEND_BLOCK_SIZE", 256)
    monkeypatch.setattr(lb, "_BACKEND_STAGE2_ERROR_FEEDBACK", False)

    with pytest.raises(RuntimeError, match="already registered with different options"):
        lb.register_lowbit_backend(bitwidth=2, error_feedback=True)


def test_register_lowbit_backend_without_extension(monkeypatch):
    monkeypatch.setattr(lb, "_REGISTERED", False)
    monkeypatch.setattr(lb, "_HAS_EXTENSION", False)
    monkeypatch.setattr(lb, "_EXTENSION_IMPORT_ERROR", RuntimeError("missing extension"))

    with pytest.raises(RuntimeError, match="extension is not available"):
        lb.register_lowbit_backend()


def test_create_lowbit_pg_without_extension(monkeypatch):
    monkeypatch.setattr(lb, "_HAS_EXTENSION", False)
    monkeypatch.setattr(lb, "_EXTENSION_IMPORT_ERROR", RuntimeError("missing extension"))

    with pytest.raises(RuntimeError, match="extension is not available"):
        lb._create_lowbit_pg(store=object(), rank=0, size=1, timeout=1)


def test_create_lowbit_pg_passes_explicit_options(monkeypatch):
    captured = {}

    def fake_create_backend(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(lb, "_HAS_EXTENSION", True)
    monkeypatch.setattr(lb, "_BACKEND_BITWIDTH", 2)
    monkeypatch.setattr(lb, "_BACKEND_ERROR_FEEDBACK", True)
    monkeypatch.setattr(lb, "_BACKEND_BLOCK_SIZE", 128)
    monkeypatch.setattr(lb, "_BACKEND_STAGE2_ERROR_FEEDBACK", True)
    monkeypatch.setattr(lb, "create_backend", fake_create_backend)

    lb._create_lowbit_pg(store="s", rank=1, size=2, timeout=3)

    assert captured["bitwidth"] == 2
    assert captured["error_feedback"] is True
    assert captured["block_size"] == 128
    assert captured["stage2_error_feedback"] is True
