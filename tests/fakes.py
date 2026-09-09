"""Hardware stand-ins shared by the server tests.

CI has no serial adapter, no I2C bus and no BNO055, so the routes that touch them
are exercised against these. They are installed at the seams the server already
has (`get_hwi`, `get_state_imu`, `hwi_instance`), never by stubbing the vendor
libraries, so a test reads like the request it makes.
"""

import tnkr_server


class FakeIO:
    """Stub servo bus. Servos whose id is in `dead` never answer."""

    def __init__(self, dead=()):
        self.DEAD = set(dead)

    def set_kps(self, ids, kps):
        if set(ids) & self.DEAD:
            raise OSError("timeout")

    def read_present_position(self, ids):
        if set(ids) & self.DEAD:
            raise OSError("timeout")
        return [0.0]

    def disable_torque(self, ids):
        pass


class FakeHWI:
    """Enough of rustypot_position_hwi.HWI for the routes under test.

    `get_present_position` mirrors the real one's contract: one joint, read-only,
    and an OSError naming the joint and id when the servo stays silent. That is
    what the walk pre-flight's roll-call relies on."""

    def __init__(self, joints=None, dead=()):
        self.joints = dict(joints) if joints is not None else dict(tnkr_server.JOINTS)
        self.low_torque_kps = [2]
        self.io = FakeIO(dead)
        self.turned_off = False
        self.joints_offsets = {name: 0.0 for name in self.joints}
        self.init_pos = {name: 0.0 for name in self.joints}

    def get_present_position(self, joint_name):
        sid = self.joints[joint_name]
        if sid in self.io.DEAD:
            raise OSError(
                f"read_present_position failed for '{joint_name}' (id {sid}) after 3 attempts: timeout"
            )
        return 0.0

    def close(self):
        pass

    def turn_off(self):
        self.turned_off = True


class FakeBNO055:
    """The already-calibrated chip behind the state IMU's handle."""

    calibration_status = (3, 3, 3, 3)
    calibrated = True
    offsets_accelerometer = (1, 2, 3)
    offsets_gyroscope = (4, 5, 6)
    offsets_magnetometer = (7, 8, 9)


class FakeStateImu:
    def __init__(self):
        self.imu = FakeBNO055()
