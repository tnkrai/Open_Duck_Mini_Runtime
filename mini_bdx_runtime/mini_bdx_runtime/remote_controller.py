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
STALE_TIMEOUT = 0.5  # seconds — zero commands if no fresh data


class RemoteController:
    def __init__(self, command_freq):
        self.command_freq = command_freq
        self.buttons = Buttons()
        self.last_commands = np.zeros(7)
        self.last_left_trigger = 0.0
        self.last_right_trigger = 0.0

    def _read_commands(self):
        """Read latest commands from the shared file."""
        try:
            with open(COMMAND_FILE, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

        # Check staleness
        ts = data.get("timestamp", 0)
        if time.time() - ts > STALE_TIMEOUT:
            return None

        return data

    def get_last_command(self):
        data = self._read_commands()

        if data is None:
            # Stale or missing — return zeros (safe stop)
            self.last_commands = np.zeros(7)
            btns = data if data else {}
            self.buttons.update(False, False, False, False, False, False, False, False)
            self.last_left_trigger = 0.0
            self.last_right_trigger = 0.0
        else:
            cmds = data.get("commands", [0.0] * 7)
            self.last_commands = np.array(cmds[:7], dtype=float)

            btns = data.get("buttons", {})
            self.buttons.update(
                btns.get("A", False),
                btns.get("B", False),
                btns.get("X", False),
                btns.get("Y", False),
                btns.get("LB", False),
                btns.get("RB", False),
                btns.get("dpad_up", False),
                btns.get("dpad_down", False),
            )

            self.last_left_trigger = float(data.get("left_trigger", 0.0))
            self.last_right_trigger = float(data.get("right_trigger", 0.0))

        return (
            self.last_commands,
            self.buttons,
            self.last_left_trigger,
            self.last_right_trigger,
        )
