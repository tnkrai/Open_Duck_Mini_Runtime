"""rustypot PoisonError / PanicException must not 500 /api/state.

PyO3 PanicException subclasses BaseException, not Exception. A poisoned
rustypot mutex therefore used to kill the ASGI task (Starlette: 'No response
returned') instead of serving the last-known pose. Concurrent /api/voltage
(pypot) vs /api/state (rustypot) on the same USB adapter is what poisons it.
"""

import threading

import pytest

from mini_bdx_runtime.duck_config import DuckConfig
from mini_bdx_runtime import rustypot_position_hwi as hwi_mod
import tnkr_server


class PanicException(BaseException):
    """Stand-in for pyo3_runtime.PanicException (not installed off-robot)."""


class OkIO:
    def read_present_position(self, ids):
        return [0.5]


def make_hwi(monkeypatch, io_factory):
    monkeypatch.setattr(hwi_mod.rustypot, "feetech", lambda port, baud: io_factory())
    cfg = DuckConfig(config_json_path="/nonexistent/duck_config.json", ignore_default=True)
    return hwi_mod.HWI(cfg, usb_port="/dev/fake")


def test_panic_exception_is_not_caught_by_except_exception():
    """The contract this bug depends on: except Exception misses rust panics."""
    caught_as_exception = False
    try:
        raise PanicException("called `Result::unwrap()` on an `Err` value: PoisonError { .. }")
    except Exception:
        caught_as_exception = True
    except PanicException:
        pass
    assert caught_as_exception is False
    assert hwi_mod.is_rust_panic(
        PanicException("called `Result::unwrap()` on an `Err` value: PoisonError { .. }")
    )


def test_get_present_positions_reopens_after_poison_panic(monkeypatch):
    calls = {"n": 0}

    class PanicThenOk:
        def read_present_position(self, ids):
            calls["n"] += 1
            if calls["n"] == 1:
                raise PanicException(
                    "called `Result::unwrap()` on an `Err` value: PoisonError { .. }"
                )
            return [0.25]

    hwi = make_hwi(monkeypatch, PanicThenOk)
    hwi._IO_RETRY_DELAY = 0
    positions = hwi.get_present_positions()
    assert positions is not None
    assert len(positions) == len(hwi.joints)
    # First call panicked; reopen + retry succeeded, then the other joints.
    assert calls["n"] == 1 + len(hwi.joints)


def test_get_present_positions_returns_none_if_poison_persists(monkeypatch):
    class AlwaysPanic:
        def read_present_position(self, ids):
            raise PanicException(
                "called `Result::unwrap()` on an `Err` value: PoisonError { .. }"
            )

    hwi = make_hwi(monkeypatch, AlwaysPanic)
    hwi._IO_RETRY_DELAY = 0
    assert hwi.get_present_positions() is None


def test_get_present_positions_holds_bus_lock(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    class SlowIO:
        def read_present_position(self, ids):
            started.set()
            assert release.wait(timeout=2)
            return [0.0]

    hwi = make_hwi(monkeypatch, SlowIO)
    t = threading.Thread(target=hwi.get_present_positions)
    t.start()
    assert started.wait(timeout=2)
    assert hwi_mod.BUS_LOCK.acquire(timeout=0.05) is False
    release.set()
    t.join(timeout=2)
    assert not t.is_alive()
    assert hwi_mod.BUS_LOCK.acquire(timeout=0.5)
    hwi_mod.BUS_LOCK.release()


def test_read_state_returns_cached_pose_on_rust_panic(client, monkeypatch):
    class BoomHWI:
        joints = {"left_knee": 23}

        def get_present_positions(self):
            raise PanicException(
                "called `Result::unwrap()` on an `Err` value: PoisonError { .. }"
            )

        def close(self):
            self.closed = True

        def turn_off(self):
            pass

    boom = BoomHWI()
    monkeypatch.setattr(tnkr_server, "hwi_instance", boom)
    monkeypatch.setattr(tnkr_server, "rehome_io", None)
    monkeypatch.setattr(tnkr_server, "is_walking", lambda: False)
    tnkr_server.last_state_joints = {"left_knee": 0.42}

    r = client.get("/api/state")
    assert r.status_code == 200
    assert r.json()["joints"] == {"left_knee": 0.42}
    # Poisoned controller was dropped so the next poll can reopen.
    assert tnkr_server.hwi_instance is None


def test_voltage_holds_bus_lock_while_releasing_hwi(client, monkeypatch, tmp_path):
    held_during_release = []

    class FakeHWI:
        joints = {"left_knee": 23}
        joints_offsets = {"left_knee": 0.0}
        init_pos = {"left_knee": -0.63}
        turned_off = False

        def turn_off(self):
            self.turned_off = True

        def close(self):
            pass

    fake = FakeHWI()
    monkeypatch.setattr(tnkr_server, "CONFIG_PATH", str(tmp_path / "duck_config.json"))
    monkeypatch.setattr(tnkr_server, "hwi_instance", fake)
    monkeypatch.setattr(tnkr_server, "stance_holding", True)

    orig = tnkr_server.release_hwi

    def wrapped(*args, **kwargs):
        held_during_release.append(hwi_mod.BUS_LOCK._is_owned())
        return orig(*args, **kwargs)

    monkeypatch.setattr(tnkr_server, "release_hwi", wrapped)
    client.get("/api/voltage")
    assert held_during_release == [True]
    assert fake.turned_off is False


# ── single-joint primitives (per-joint calibration) ──────────────────────────
# get_present_positions() returns None if ANY of the fourteen fails, which is right
# for the walk loop and wrong for anything working on one joint: a silent joint 3 made
# a read for joint 7 fail with no way to name the joint that was actually quiet.


def test_get_present_position_reads_one_joint_and_subtracts_its_offset(monkeypatch):
    class OneJoint:
        def __init__(self):
            self.asked = []

        def read_present_position(self, ids):
            self.asked.append(list(ids))
            return [0.5]

    io = OneJoint()
    hwi = make_hwi(monkeypatch, lambda: io)
    hwi.joints_offsets["left_knee"] = 0.15

    assert hwi.get_present_position("left_knee") == pytest.approx(0.35)
    # exactly one servo asked, and it is the one requested
    assert io.asked == [[hwi.joints["left_knee"]]]


def test_get_present_position_raises_naming_the_joint(monkeypatch):
    """The whole point: the failure has to be attributable. _io_retry's message
    carries the joint name and its id, which is what the caller turns into copy."""

    class Dead:
        def read_present_position(self, ids):
            raise OSError("timeout")

    hwi = make_hwi(monkeypatch, Dead)
    with pytest.raises(OSError) as exc:
        hwi.get_present_position("left_knee")
    assert "left_knee" in str(exc.value)
    assert str(hwi.joints["left_knee"]) in str(exc.value)


def test_get_present_position_tolerates_a_partial_offsets_dict(monkeypatch):
    """A hand-edited duck_config.json can omit joints. A KeyError here would read as
    a dead servo, so a missing offset is 0.0."""

    class OkOne:
        def read_present_position(self, ids):
            return [0.25]

    hwi = make_hwi(monkeypatch, OkOne)
    del hwi.joints_offsets["left_knee"]
    assert hwi.get_present_position("left_knee") == pytest.approx(0.25)


def test_set_joint_torque_retries_a_transient_failure(monkeypatch):
    """Reaching through to hwi.io.disable_torque() directly skipped _io_retry, so a
    one-off timeout was a hard failure instead of one of three attempts."""
    calls = {"n": 0}

    class FlakyTorque:
        def read_present_position(self, ids):
            return [0.0]

        def disable_torque(self, ids):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("timeout")

    hwi = make_hwi(monkeypatch, FlakyTorque)
    hwi.set_joint_torque("left_knee", False)
    assert calls["n"] == 2


def test_set_joint_torque_normalises_a_rust_panic(monkeypatch):
    """PanicException subclasses BaseException, so a raw call let it escape every
    `except Exception` and kill the ASGI task. _io_retry turns it into OSError."""

    class PanickingTorque:
        def read_present_position(self, ids):
            return [0.0]

        def disable_torque(self, ids):
            raise PanicException(
                "called `Result::unwrap()` on an `Err` value: PoisonError { .. }"
            )

    hwi = make_hwi(monkeypatch, PanickingTorque)
    with pytest.raises(OSError):
        hwi.set_joint_torque("left_knee", False)


def test_set_joint_torque_enables_only_the_named_joint(monkeypatch):
    class Recording:
        def __init__(self):
            self.enabled = []

        def read_present_position(self, ids):
            return [0.0]

        def enable_torque(self, ids):
            self.enabled.append(list(ids))

    io = Recording()
    hwi = make_hwi(monkeypatch, lambda: io)
    hwi.set_joint_torque("right_knee", True)
    assert io.enabled == [[hwi.joints["right_knee"]]]
