"""find_servo_adapter(): vendor-id matching and chip identification."""

import types

import pytest

from mini_bdx_runtime import rustypot_position_hwi as hwi_mod


def fake_port(device, vid):
    return types.SimpleNamespace(device=device, vid=vid)


def test_finds_ch343_adapter(monkeypatch):
    monkeypatch.setattr(
        hwi_mod.serial.tools.list_ports, "comports",
        lambda: [fake_port("/dev/ttyACM0", 0x1A86)],
    )
    assert hwi_mod.find_servo_adapter() == ("/dev/ttyACM0", "CH343")


def test_finds_ftdi_adapter(monkeypatch):
    monkeypatch.setattr(
        hwi_mod.serial.tools.list_ports, "comports",
        lambda: [fake_port("/dev/ttyUSB0", 0x0403)],
    )
    assert hwi_mod.find_servo_adapter() == ("/dev/ttyUSB0", "FTDI")


def test_multiple_adapters_uses_first(monkeypatch, capsys):
    monkeypatch.setattr(
        hwi_mod.serial.tools.list_ports, "comports",
        lambda: [fake_port("/dev/ttyACM0", 0x1A86), fake_port("/dev/ttyUSB0", 0x0403)],
    )
    assert hwi_mod.find_servo_adapter() == ("/dev/ttyACM0", "CH343")
    assert "multiple servo adapters" in capsys.readouterr().out


def test_no_adapter_raises_with_seen_devices(monkeypatch):
    monkeypatch.setattr(
        hwi_mod.serial.tools.list_ports, "comports",
        lambda: [fake_port("/dev/ttyAMA0", 0x1234)],
    )
    with pytest.raises(RuntimeError, match="No servo-bus USB adapter found"):
        hwi_mod.find_servo_adapter()


def test_find_servo_port_back_compat_returns_device_only(monkeypatch):
    monkeypatch.setattr(
        hwi_mod.serial.tools.list_ports, "comports",
        lambda: [fake_port("/dev/ttyACM1", 0x1A86)],
    )
    assert hwi_mod.find_servo_port() == "/dev/ttyACM1"
