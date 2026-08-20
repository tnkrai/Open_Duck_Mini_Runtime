import json
import math
from typing import Optional
import os

from mini_bdx_runtime import envelope

HOME_DIR = os.path.expanduser("~")


class DuckConfig:

    def __init__(
        self,
        config_json_path: Optional[str] = f"{HOME_DIR}/duck_config.json",
        ignore_default: bool = False,
    ):
        """
        Looks for duck_config.json in the home directory by default.
        If not found, uses default values.
        """
        self.default = False
        try:
            self.json_config = (
                json.load(open(config_json_path, "r")) if config_json_path else {}
            )
        except FileNotFoundError:
            print(
                f"Warning : didn't find the config json file at {config_json_path}, using default values"
            )
            self.json_config = {}
            self.default = True

        if config_json_path is None:
            print("Warning : didn't provide a config json path, using default values")
            self.default = True

        if self.default and not ignore_default:
            print("")
            print("")
            print("")
            print("")
            print("======")
            print(
                "WARNING : Running with default values probably won't work well. Please make a duck_config.json file and set the parameters."
            )
            res = input("Do you still want to run ? (y/N)")
            if res.lower() != "y":
                print("Exiting...")
                exit(1)

        self.start_paused = self.json_config.get("start_paused", False)
        self.imu_upside_down = self.json_config.get("imu_upside_down", False)
        self.phase_frequency_factor_offset = self.json_config.get(
            "phase_frequency_factor_offset", 0.0
        )

        # ── Safety-envelope abort thresholds ──────────────────────────────────
        # Only read when a CUSTOM policy is running (amendment A8); the built-in
        # policy never arms the guards. Both aborts cut torque and exit with a
        # named reason:
        #
        #   tilt_limit_deg        pitch or roll past this is a fall, not a gait.
        #                         Sustained for tilt_abort_ticks ticks -> abort.
        #                         An unreadable IMU counts as over the limit.
        #   tilt_abort_ticks      consecutive bad ticks before the tilt abort
        #                         fires. 8 at 50 Hz = 0.16 s, enough to ride out
        #                         a stumble.
        #   budget_overrun_ticks  consecutive ticks over 1/control_freq before
        #                         the budget abort fires. 10 at 50 Hz = 0.2 s.
        #
        # A missing key uses the default. A key set to something nonsensical
        # (zero ticks, a negative limit, a string) ALSO uses the default and says
        # so: a config edit must never be able to leave a custom policy running
        # with no guard at all.
        self.tilt_limit_deg = self._number(
            "tilt_limit_deg", envelope.DEFAULT_TILT_LIMIT_DEG, low=1.0, high=90.0
        )
        self.tilt_abort_ticks = self._positive_int(
            "tilt_abort_ticks", envelope.DEFAULT_TILT_ABORT_TICKS
        )
        self.budget_overrun_ticks = self._positive_int(
            "budget_overrun_ticks", envelope.DEFAULT_BUDGET_OVERRUN_TICKS
        )

        expression_features = self.json_config.get("expression_features", {})

        self.eyes = expression_features.get("eyes", False)
        self.projector = expression_features.get("projector", False)
        self.antennas = expression_features.get("antennas", False)
        self.speaker = expression_features.get("speaker", False)
        self.microphone = expression_features.get("microphone", False)
        self.camera = expression_features.get("camera", False)

        # default joints offsets are 0.0
        self.joints_offset = self.json_config.get(
            "joints_offsets",
            {
                "left_hip_yaw": 0.0,
                "left_hip_roll": 0.0,
                "left_hip_pitch": 0.0,
                "left_knee": 0.0,
                "left_ankle": 0.0,
                "neck_pitch": 0.0,
                "head_pitch": 0.0,
                "head_yaw": 0.0,
                "head_roll": 0.00,
                "right_hip_yaw": 0.0,
                "right_hip_roll": 0.0,
                "right_hip_pitch": 0.0,
                "right_knee": 0.0,
                "right_ankle": 0.0,
            },
        )

    # ── Typed reads ───────────────────────────────────────────────────────────
    # Plain .get() is fine for a flag, but a threshold that arms a safety guard
    # has to survive a hand-edited json file. A bad value falls back and says so
    # on stdout rather than silently disarming the guard it configures.

    def _number(self, key, default, low=None, high=None):
        raw = self.json_config.get(key, default)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            print(f"Warning : {key}={raw!r} is not a number, using {default}")
            return float(default)
        if not math.isfinite(value):
            print(f"Warning : {key}={raw!r} is not finite, using {default}")
            return float(default)
        if (low is not None and value < low) or (high is not None and value > high):
            print(
                f"Warning : {key}={value} is outside [{low}, {high}], "
                f"using {default}"
            )
            return float(default)
        return value

    def _positive_int(self, key, default):
        raw = self.json_config.get(key, default)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            print(f"Warning : {key}={raw!r} is not a whole number, using {default}")
            return int(default)
        if value < 1:
            print(f"Warning : {key}={value} must be at least 1, using {default}")
            return int(default)
        return value
