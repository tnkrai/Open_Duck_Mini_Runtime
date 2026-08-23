"""Walk input flags, pad silence, and agent pad start (no hardware)."""

import pytest

from mini_bdx_runtime.pad import (
    is_controller_device,
    is_xbox_name,
    should_zero_commands,
    valid_address,
    walk_flags,
)

tnkr_server = pytest.importorskip("tnkr_server")


def test_walk_flags_keyboard_keeps_remote():
    assert "--remote" in walk_flags("keyboard")
    assert "--remote" in walk_flags("")
    assert "--commands" in walk_flags("keyboard")


def test_walk_flags_pad_omits_remote():
    flags = walk_flags("pad")
    assert "--remote" not in flags
    assert "--commands" in flags


def test_silence_zeros_after_300ms():
    assert should_zero_commands(True, 0.0, 1.0) is False
    assert should_zero_commands(False, 0.0, 0.2) is False
    assert should_zero_commands(False, 0.0, 0.3) is True


def test_xbox_name_and_mac():
    assert is_xbox_name("Xbox Wireless Controller")
    assert not is_xbox_name("AirPods")
    assert valid_address("AA:BB:CC:DD:EE:FF")
    assert not valid_address("not-a-mac")


def test_controller_matches_icon_before_name():
    # During scan the MAC shows first; Icon: input-gaming arrives before Name.
    assert is_controller_device(name="B8:41:76:CD:53:D3", icon="input-gaming")
    assert not is_controller_device(name="AirPods", icon="audio-card")


def test_agent_pad_start_without_joystick_is_409_and_does_not_spawn(
    client, fake_walk_dir, monkeypatch
):
    from conftest import write_walk_script

    write_walk_script(fake_walk_dir, "import time; time.sleep(30)\n")
    monkeypatch.setattr(tnkr_server, "joystick_present", lambda: False)
    spawned = []
    monkeypatch.setattr(
        tnkr_server.subprocess,
        "Popen",
        lambda *a, **k: spawned.append(a) or (_ for _ in ()).throw(AssertionError("spawned")),
    )
    res = client.post("/api/walk/start", json={"input": "pad"})
    assert res.status_code == 409
    assert "PAD_NOT_FOUND" in res.json()["detail"]
    assert spawned == []


def test_agent_keyboard_start_argv_has_remote(client, fake_walk_dir, monkeypatch):
    from conftest import write_walk_script

    write_walk_script(fake_walk_dir, "import time; time.sleep(30)\n")
    seen = {}
    real = tnkr_server.subprocess.Popen

    def capturing(cmd, **kwargs):
        seen["cmd"] = cmd
        return real(cmd, **kwargs)

    monkeypatch.setattr(tnkr_server.subprocess, "Popen", capturing)
    res = client.post("/api/walk/start", json={"input": "keyboard"})
    assert res.status_code == 200
    assert "--remote" in seen["cmd"]
    client.post("/api/walk/stop")


def test_agent_pad_start_argv_has_no_remote(client, fake_walk_dir, monkeypatch):
    from conftest import write_walk_script

    write_walk_script(fake_walk_dir, "import time; time.sleep(30)\n")
    seen = {}
    real = tnkr_server.subprocess.Popen

    def capturing(cmd, **kwargs):
        seen["cmd"] = cmd
        return real(cmd, **kwargs)

    monkeypatch.setattr(tnkr_server, "joystick_present", lambda: True)
    monkeypatch.setattr(tnkr_server.subprocess, "Popen", capturing)
    res = client.post("/api/walk/start", json={"input": "pad"})
    assert res.status_code == 200
    assert "--remote" not in seen["cmd"]
    client.post("/api/walk/stop")
