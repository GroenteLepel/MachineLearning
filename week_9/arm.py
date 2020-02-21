import numpy as np
from functools import partial
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


class Arm:
    def __init__(self, n_joints: int, noise_parameter: float, angles=None):
        self.n_joints = n_joints
        if angles is None:
            self.joint_angle = np.zeros(n_joints)
        else:
            self.joint_angle = angles
        self.noise_parameter = noise_parameter

    def move_to(self, target, time_window):
        pass

    def _calc_diff_angle(self, action, time_step_size):
        xi = self.noise_parameter * time_step_size
        noise = np.random.normal(size=self.n_joints,
                                 scale=np.sqrt(xi))
        return action * time_step_size + noise

    def increment_angles(self, action, time_step_size):
        self.joint_angle += self._calc_diff_angle(action, time_step_size)

    def draw(self, dashed: bool = False):
        if not plt.get_fignums():
            size = 5
            fig = plt.figure(figsize=(size, size))
            ax = fig.add_subplot(1, 1, 1)
        else:
            fig = plt.gcf()
            ax = plt.gca()

        x_individual = np.cos(self.joint_angle)
        y_individual = np.sin(self.joint_angle)
        x_joints, y_joints = np.zeros(self.n_joints + 1), np.zeros(
            self.n_joints + 1)
        for i in range(1, self.n_joints + 1):
            x_joints[i] = x_individual[:i].sum()
            y_joints[i] = y_individual[:i].sum()

        ax.grid(ls='--')
        axes_range = - self.n_joints - 1, self.n_joints + 1
        ax.set_xticks(np.arange(axes_range[0], axes_range[1]))
        ax.set_yticks(np.arange(axes_range[0], axes_range[1]))
        ax.set_xlim(axes_range)
        ax.set_ylim(axes_range)
        if dashed:
            ax.plot(x_joints, y_joints, marker='o', ls='--')
        else:
            ax.plot(x_joints, y_joints, marker='o')
