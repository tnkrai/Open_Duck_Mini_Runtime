"""The Bluetooth radio itself: is it there, is it on, and do we say so.

These exist because of a real support case. An operator spent half an hour on
Studio's pair screen with a Pi whose adapter was powered off. `bluetoothctl
scan` cheerfully returned an empty list, Studio rendered "turn the controller
on, then hold sync", and nothing anywhere -- agent, API, analytics -- recorded
that the radio was down.

The agent telemetry later showed `POST /api/pad/scan` returning 200 forty-seven
times, so the wake DID run and the adapter stayed off anyway. It ran
`bluetoothctl power on`, discarded the result and returned None, so which of the
three possible causes it hit is unknowable after the fact. Every test below pins
one link in that chain, and the escalation tests pin the specific thing the old
code could not do: notice that the wake did not take.

The ladder now runs only from `wake_adapter()`, which only the Turn on route
calls. The last group of tests pins that: a scan, a pair, a disconnect and a
forget all read the radio and refuse, and none of them powers anything on.
"""

import pytest

from mini_bdx_runtime import pad


class FakeRun:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


SHOW_ON = "Controller AA:BB:CC:DD:EE:FF\n\tPowered: yes\n\tDiscoverable: no\n"
SHOW_OFF = "Controller AA:BB:CC:DD:EE:FF\n\tPowered: no\n\tDiscoverable: no\n"
SHOW_NONE = "No default controller available\n"
RFKILL_CLEAR = "0: hci0: Bluetooth\n\tSoft blocked: no\n\tHard blocked: no\n"
RFKILL_SOFT = "0: hci0: Bluetooth\n\tSoft blocked: yes\n\tHard blocked: no\n"
RFKILL_HARD = "0: hci0: Bluetooth\n\tSoft blocked: no\n\tHard blocked: yes\n"

BLUEZ_REFUSAL = "Failed to set power on: org.bluez.Error.Blocked"


@pytest.fixture
def radio(monkeypatch):
    """A fake Pi radio.

    Three switches decide which wake step is allowed to succeed, so a test can
    say "the piped call does nothing but the pty works" -- which is the shape
    the support case most likely had.
    """
    state = {
        "show": SHOW_ON,
        "rfkill": RFKILL_CLEAR,
        "calls": [],
        "unblock_works": True,
        "piped_works": True,
        "pty_works": True,
    }

    def fake_bt(*args, timeout=12.0):
        state["calls"].append(("bluetoothctl",) + args)
        if args and args[0] == "show":
            return FakeRun(state["show"])
        if args[:2] == ("power", "on"):
            if not state["piped_works"]:
                return FakeRun("", returncode=1, stderr=BLUEZ_REFUSAL)
            if not state["rfkill"].count("Soft blocked: yes"):
                state["show"] = SHOW_ON
            return FakeRun("Changing power on succeeded")
        return FakeRun("")

    def fake_rfkill(*args, timeout=5.0):
        state["calls"].append(("rfkill",) + args)
        if args and args[0] == "list":
            return FakeRun(state["rfkill"])
        if args and args[0] == "unblock":
            if not state["unblock_works"]:
                return FakeRun("", returncode=1, stderr="Operation not permitted")
            state["rfkill"] = RFKILL_CLEAR
            state["show"] = SHOW_OFF
            return FakeRun("")
        return FakeRun("")

    def fake_pty():
        state["calls"].append(("pty", "power on"))
        if not state["pty_works"]:
            return False, BLUEZ_REFUSAL
        state["show"] = SHOW_ON
        return True, ""

    monkeypatch.setattr(pad, "_bt", fake_bt)
    monkeypatch.setattr(pad, "_rfkill", fake_rfkill)
    monkeypatch.setattr(pad, "_power_on_via_pty", fake_pty)
    monkeypatch.setattr(pad, "list_xbox_devices", lambda: [])
    monkeypatch.setattr(pad.time, "sleep", lambda _s: None)
    return state


def used(state, *call):
    return call in state["calls"]


# -- reading the state -------------------------------------------------------


def test_powered_adapter_reports_no_reason(radio):
    state = pad.adapter_state()
    assert state["present"] is True
    assert state["powered"] is True
    assert state["blocked"] is False
    assert state["reason"] == pad.ADAPTER_OK
    # a plain read attempts nothing, so it claims nothing
    assert state["wokeVia"] is None and state["wakeError"] is None


def test_powered_off_is_off_not_missing(radio):
    radio["show"] = SHOW_OFF
    assert pad.adapter_state()["reason"] == pad.ADAPTER_OFF


def test_absent_controller_is_missing(radio):
    radio["show"] = SHOW_NONE
    state = pad.adapter_state()
    assert state["present"] is False
    assert state["reason"] == pad.ADAPTER_MISSING


def test_soft_block_outranks_missing(radio):
    # A blocked adapter is not published as a controller, so `show` says
    # "No default controller available" -- which reads as missing hardware
    # unless rfkill is consulted first. It is the block that is actionable.
    radio["show"] = SHOW_NONE
    radio["rfkill"] = RFKILL_SOFT
    assert pad.adapter_state()["reason"] == pad.ADAPTER_BLOCKED


def test_hard_block_outranks_soft(radio):
    radio["rfkill"] = RFKILL_HARD
    radio["show"] = SHOW_NONE
    state = pad.adapter_state()
    assert state["reason"] == pad.ADAPTER_HARD_BLOCKED
    assert state["hardBlocked"] is True


def test_missing_rfkill_binary_is_not_a_block(monkeypatch, radio):
    monkeypatch.setattr(pad, "_rfkill", lambda *a, **k: None)
    assert pad.adapter_state()["reason"] == pad.ADAPTER_OK


# -- the escalation ladder ---------------------------------------------------


def test_healthy_radio_is_left_alone(radio):
    state = pad.wake_adapter()
    assert state["wokeVia"] == "already_on"
    assert not used(radio, "bluetoothctl", "power", "on")
    assert not used(radio, "pty", "power on")


def test_soft_block_is_cleared_by_rfkill_and_named(radio):
    radio["rfkill"] = RFKILL_SOFT
    radio["show"] = SHOW_NONE
    state = pad.wake_adapter()
    assert used(radio, "rfkill", "unblock", "bluetooth")
    # the fake leaves it merely off, so the piped call is what finishes
    assert state["wokeVia"] == "bluetoothctl"
    assert state["blocked"] is False


def test_piped_call_alone_is_enough_when_it_works(radio):
    radio["show"] = SHOW_OFF
    state = pad.wake_adapter()
    assert state["wokeVia"] == "bluetoothctl"
    assert not used(radio, "pty", "power on"), "no need to escalate past a working call"


def test_pty_rescues_a_piped_call_that_does_nothing(radio):
    # The likely shape of the support case: the operator typed `power on` into
    # an interactive bluetoothctl and it worked first try, while the agent's
    # piped call had already failed dozens of times.
    radio["show"] = SHOW_OFF
    radio["piped_works"] = False
    state = pad.wake_adapter()
    assert used(radio, "bluetoothctl", "power", "on")
    assert used(radio, "pty", "power on")
    assert state["wokeVia"] == "pty"
    assert state["reason"] == pad.ADAPTER_OK
    assert state["wakeError"] is None


def test_total_failure_keeps_bluez_own_sentence(radio):
    radio["show"] = SHOW_OFF
    radio["piped_works"] = False
    radio["pty_works"] = False
    state = pad.wake_adapter()
    assert state["wokeVia"] is None
    assert state["reason"] == pad.ADAPTER_OFF
    assert BLUEZ_REFUSAL in state["wakeError"], "the one string that explains the failure"


def test_refused_unblock_is_reported_not_swallowed(radio):
    radio["rfkill"] = RFKILL_SOFT
    radio["show"] = SHOW_NONE
    radio["unblock_works"] = False
    radio["piped_works"] = False
    radio["pty_works"] = False
    state = pad.wake_adapter()
    assert "rfkill" in state["wakeError"]


def test_hard_block_does_not_pretend_to_fix_itself(radio):
    radio["rfkill"] = RFKILL_HARD
    state = pad.wake_adapter()
    assert state["reason"] == pad.ADAPTER_HARD_BLOCKED
    assert not used(radio, "rfkill", "unblock", "bluetooth")
    assert not used(radio, "bluetoothctl", "power", "on")
    assert not used(radio, "pty", "power on")
    assert "hard blocked" in state["wakeError"]


# -- what the API returns ----------------------------------------------------


def test_status_carries_the_adapter_even_with_no_devices(radio):
    status = pad.pad_status()
    assert status["devices"] == []
    assert status["adapter"]["reason"] == pad.ADAPTER_OK


def test_scan_skips_the_18_second_scan_when_the_radio_is_down(radio):
    radio["show"] = SHOW_OFF
    radio["piped_works"] = False
    radio["pty_works"] = False
    status = pad.scan_pad()
    assert not any(
        c[:2] == ("bluetoothctl", "--timeout") for c in radio["calls"]
    ), "scanning a dead radio burns 18s to learn nothing"
    assert status["adapter"]["reason"] == pad.ADAPTER_OFF
    assert status["present"] is False


def test_scan_runs_normally_once_the_radio_is_up(radio):
    # Was: set the adapter OFF and assert the scan ran anyway, which asserted
    # that scanning powers the radio on. It does not any more, so the operator's
    # press has to happen first -- and then the scan behaves exactly as before.
    radio["show"] = SHOW_OFF
    assert pad.wake_adapter()["reason"] is pad.ADAPTER_OK
    pad.scan_pad(timeout=1.0)
    assert any(c[:2] == ("bluetoothctl", "--timeout") for c in radio["calls"])


@pytest.mark.parametrize(
    "call",
    [
        lambda: pad.scan_pad(timeout=1.0),
        lambda: pad.pair_pad("AA:BB:CC:DD:EE:FF"),
        lambda: pad.disconnect_pad("AA:BB:CC:DD:EE:FF"),
        lambda: pad.forget_pad("AA:BB:CC:DD:EE:FF"),
    ],
    ids=["scan", "pair", "disconnect", "forget"],
)
def test_no_pad_call_powers_the_radio_on_by_itself(radio, call):
    """The whole point of the change: nothing wakes the radio behind the operator.

    A radio that powers itself up cannot be reported. The screen said "hold
    sync" for half an hour precisely because the wake was a side effect nobody
    could see, so every one of these has to read the state and stop.
    """
    radio["show"] = SHOW_OFF
    status = call()
    assert status["adapter"]["reason"] == pad.ADAPTER_OFF
    assert not any(c[:2] == ("bluetoothctl", "power") for c in radio["calls"])
    assert not any(c[0] == "rfkill" and c[1] == "unblock" for c in radio["calls"])
    assert ("pty", "power on") not in radio["calls"]


def test_the_wake_is_the_one_thing_that_powers_the_radio_on(radio):
    radio["show"] = SHOW_OFF
    state = pad.wake_adapter()
    assert state["reason"] is pad.ADAPTER_OK
    assert state["wokeVia"] == "bluetoothctl"


def test_pair_refuses_early_when_the_radio_is_down(radio):
    radio["show"] = SHOW_NONE
    radio["rfkill"] = RFKILL_HARD
    status = pad.pair_pad("AA:BB:CC:DD:EE:FF")
    assert status["connected"] is False
    assert status["adapter"]["reason"] == pad.ADAPTER_HARD_BLOCKED


def test_pad_status_reuses_a_supplied_adapter_reading(radio):
    supplied = pad._adapter_dict(present=True, powered=True, soft=False, hard=False)
    radio["calls"].clear()
    pad.pad_status(supplied)
    assert not any(c[:2] == ("bluetoothctl", "show") for c in radio["calls"])


# -- the parser that turns BlueZ noise into one sentence ---------------------


def test_first_failure_picks_the_refusal_out_of_ansi_noise():
    text = "\x1b[0;94m[bluetooth]\x1b[0m# power on\r\n" + BLUEZ_REFUSAL + "\r\n"
    assert pad._first_failure(text) == BLUEZ_REFUSAL


def test_first_failure_is_empty_when_nothing_failed():
    assert pad._first_failure("Changing power on succeeded\n") == ""


def test_first_failure_is_clipped():
    assert len(pad._first_failure("Failed: " + "x" * 500)) <= pad._WAKE_ERROR_MAX
