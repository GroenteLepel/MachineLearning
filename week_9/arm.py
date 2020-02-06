import numpy as np
from functools import partial
from scipy.optimize import fsolve

from week_9.joint import Joint


class Arm:
    def __init__(self, n_joints: int, noise: float):
        self.n_joints = n_joints
        self.joint_angle = np.zeros(n_joints)
        self.noise = noise

    def _gen_initial_joints(self):
        return [Joint(0) for _ in self.n_joints]

    def move_to(self, target, time_window):
        time_step = 1
        steps = np.arange(time_window, time_step)
        for t in steps:
            diff_angle = self._calc_diff_angle(time_window, time_step, t)
            self.joint_angle += diff_angle

    def _calc_diff_angle(self, time_window, time_step, time):
        action = self._calc_action(time_window, time)

        noise = np.sqrt(self.noise * time_step) * np.random.normal()
        return action * time_step + noise

    def _calc_action(self, time_window, time):
        expected_angle = self._calc_expected_angle()

        return 1 / (time_window - time) * (expected_angle - self.joint_angle)

    def _calc_expected_angle(self):
        pass

    def expected_angle(self, alpha, time_window):
        pass
