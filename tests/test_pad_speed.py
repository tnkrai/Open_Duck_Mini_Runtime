"""What a pad operation is allowed to cost.

Measured on a real Pi Zero 2W: a disconnect took 7.77s, of which 2.88s was BlueZ
genuinely tearing down the HID link and 3.8s was `bluetoothctl info` spawns for
devices that were then filtered out for not being controllers. The duck had nine
bonded devices; eight were phone and laptop leftovers. That cost grew with every
device the radio had ever seen, which is not something an operator can be asked
to know about, let alone prune by hand.

These tests pin the three things that made it O(1) instead. They are about cost,
so they assert on the calls made, not on wall-clock, which would be flaky.
"""

from __future__ import annotations

import threading

import pytest

from mini_bdx_runtime import pad

PAD = "B8:41:76:CD:53:D3"
#: The shape of a real radio: one controller, and the random-static BLE addresses
#: a phone and a laptop leave behind.
JUNK = [
    "6B:31:C7:80:59:64",
    "52:8B:F8:3A:8A:B2",
    "66:0E:9C:CB:C4:3D",
    "6B:85:CA:F1:9C:52",
    "5A:66:02:D6:A1:A8",
    "6B:E1:45:72:F6:DE",
    "48:C4:BA:3A:19:E6",
    "DC:68:80:48:52:CE",
]


class Run:
    def __init__(self, stdout=""):
        self.stdout, self.stderr, self.returncode = stdout, "", 0


@pytest.fixture
def radio(monkeypatch):
    """A cluttered but healthy radio, and a record of what was asked of it."""
    calls: list[tuple[str, ...]] = []
    # Two `info` reads have to be in flight at the same time for this to pass.
    # Serial code sits at the first wait until the barrier times out and breaks,
    # which fails the assertion rather than hanging the suite. Counting distinct
    # thread idents looked simpler and was flaky: the fake returns instantly, so
    # one worker can drain the whole queue before a second is ever spawned.
    gate = threading.Barrier(2, timeout=2.0)
    concurrent = threading.Semaphore(2)

    listing = "".join(f"Device {a} {a.replace(':', '-')}\n" for a in JUNK)
    listing += f"Device {PAD} Xbox Wireless Controller\n"

    def fake_bt(*args, timeout=12.0):
        calls.append(args)
        if args[0] == "show":
            return Run("Controller %s\n\tPowered: yes\n" % PAD)
        if args[0] == "devices":
            # `devices Paired` returns only the bonded ones. The junk is cached,
            # not bonded, which is what the trace off the real duck showed: nine
            # info reads before `devices Paired` and none after it.
            return Run(f"Device {PAD} Xbox Wireless Controller\n" if len(args) > 1 else listing)
        if args[0] == "info":
            if concurrent.acquire(blocking=False):
                try:
                    gate.wait()
                except threading.BrokenBarrierError:
                    pass
            if args[1] == PAD:
                return Run(
                    f"Device {PAD}\n\tName: Xbox Wireless Controller\n"
                    "\tIcon: input-gaming\n\tPaired: yes\n\tConnected: yes\n"
                )
            return Run(f"Device {args[1]}\n\tPaired: no\n\tConnected: no\n")
        return Run("")

    monkeypatch.setattr(pad, "_bt", fake_bt)
    monkeypatch.setattr(
        pad, "_rfkill", lambda *a, **k: Run("0: hci0: Bluetooth\n\tSoft blocked: no\n\tHard blocked: no\n")
    )
    monkeypatch.setattr(pad.time, "sleep", lambda _s: None)
    monkeypatch.setattr(pad, "joystick_present", lambda: True)
    return {"calls": calls, "gate": gate}


def counted(radio, verb: str) -> int:
    return sum(1 for c in radio["calls"] if c and c[0] == verb)


def test_a_cluttered_radio_is_read_in_parallel(radio):
    """Nine devices, nine `info` reads, not nine round trips.

    Serial, this was the single largest cost in the pad surface, and it scaled
    with how many devices the radio had ever noticed rather than how many
    controllers exist.
    """
    devices = pad.list_xbox_devices()
    assert [d["address"] for d in devices] == [PAD], "only the controller survives the filter"
    assert counted(radio, "info") == len(JUNK) + 1, "every device is still considered"
    assert not radio["gate"].broken, "the reads were issued one after another"


def test_a_known_address_does_not_enumerate_the_whole_radio(radio):
    """Every press in Studio comes off a device row, so the address is known.

    Enumerating here only to return a status the caller re-reads at the end was
    a whole extra pass over every device on the radio.
    """
    pad.disconnect_pad(PAD)
    assert counted(radio, "devices") == 2, "one enumeration: the status that is returned"
    # `devices` and `devices Paired` are the two halves of ONE enumeration.
    assert [c for c in radio["calls"] if c and c[0] == "devices"] == [
        ("devices",),
        ("devices", "Paired"),
    ]


def test_disconnect_still_disconnects(radio):
    """The cost came off, the behaviour did not."""
    status = pad.disconnect_pad(PAD)
    assert ("disconnect", PAD) in radio["calls"]
    assert status["devices"][0]["address"] == PAD


def test_a_joystick_is_found_from_its_device_node(monkeypatch):
    """`import pygame; pygame.init()` costs ~2s on a Pi Zero 2W. A bonded pad
    always has a /dev/input/js node, so the slow path is only for proving a
    negative -- where two seconds costs nobody anything."""
    seen: list[str] = []

    def fake_glob(pattern):
        seen.append(pattern)
        return ["/dev/input/js0"]

    monkeypatch.setattr(pad.glob, "glob", fake_glob)
    assert pad.joystick_present() is True
    assert seen == ["/dev/input/js*"], "the device node is checked before pygame"


def test_no_joystick_still_falls_back_to_pygame(monkeypatch):
    """A controller with no js node has to answer the way it always did."""
    monkeypatch.setattr(pad.glob, "glob", lambda _p: [])
    asked = []

    class FakePygame:
        class joystick:
            @staticmethod
            def init():
                pass

            @staticmethod
            def get_count():
                asked.append(True)
                return 1

        @staticmethod
        def init():
            pass

    monkeypatch.setitem(__import__("sys").modules, "pygame", FakePygame)
    assert pad.joystick_present() is True
    assert asked, "pygame was never consulted"
