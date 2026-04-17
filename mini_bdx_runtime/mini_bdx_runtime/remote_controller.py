"""
Remote controller for TNKR dashboard keyboard control.

Drop-in replacement for XBoxController — reads commands from a shared file
written by the TNKR HTTP server. Same get_last_command() interface so the
walk script doesn't need to change its main loop.
"""

import json
import time

import numpy as np
from mini_bdx_runtime.buttons import Buttons

COMMAND_FILE = "/dev/shm/tnkr_remote_commands.json"
# Zero commands if no NEW data arrives within this window.
# Must be long enough to survive HTTP round-trip jitter.
NO_UPDATE_TIMEOUT = 2.0


class RemoteController:
    def __init__(self, command_freq):
        self.command_freq = command_freq
        self.buttons = Buttons()
        self.last_commands = np.zeros(7)
        self.last_left_trigger = 0.0
        self.last_right_trigger = 0.0
        self._last_seen_ts = 0.0  # track last file timestamp we processed
        self._last_update_time = time.time()  # wall clock of last new data
        self._last_btns = {}  # last known button state for edge detection

    def get_last_command(self):
        try:
            with open(COMMAND_FILE, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = None

        if data is not None:
            file_ts = data.get("timestamp", 0)
            if file_ts != self._last_seen_ts:
                # New data arrived — update everything
                self._last_seen_ts = file_ts
                self._last_update_time = time.time()

                cmds = data.get("commands", [0.0] * 7)
                self.last_commands = np.array(cmds[:7], dtype=float)

                self._last_btns = data.get("buttons", {})
                self.last_left_trigger = float(data.get("left_trigger", 0.0))
                self.last_right_trigger = float(data.get("right_trigger", 0.0))

        # Always update buttons with last known state so edge detection
        # correctly resets triggered=False on subsequent reads
        self.buttons.update(
            self._last_btns.get("A", False),
            self._last_btns.get("B", False),
            self._last_btns.get("X", False),
            self._last_btns.get("Y", False),
            self._last_btns.get("LB", False),
            self._last_btns.get("RB", False),
            self._last_btns.get("dpad_up", False),
            self._last_btns.get("dpad_down", False),
        )

        # If no new data for too long, zero out (safe stop)
        if time.time() - self._last_update_time > NO_UPDATE_TIMEOUT:
            self.last_commands = np.zeros(7)
            self.last_left_trigger = 0.0
            self.last_right_trigger = 0.0

        return (
            self.last_commands,
            self.buttons,
            self.last_left_trigger,
            self.last_right_trigger,
        )
