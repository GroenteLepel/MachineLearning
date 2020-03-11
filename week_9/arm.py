import numpy as np
import matplotlib.pyplot as plt


class Arm:
    def __init__(self, n_joints: int, noise_parameter: float, angles=None):
        self.n_joints = n_joints
        if angles is None:
            self.joint_angle = np.random.normal(size=n_joints)
        else:
            self.joint_angle = angles
        self.noise_parameter = noise_parameter

    def _calc_diff_angle(self, action, time_step_size):
        """
        Calculates the dtheta as from equation 2 in the report.
        :param action: action calculated using equation 3 in report.
        :param time_step_size: float, dt value.
        :return: ndarray, dtype=float.
        """
        xi = self.noise_parameter * time_step_size
        noise = np.random.normal(size=self.n_joints,
                                 scale=np.sqrt(xi))
        return action * time_step_size + noise

    def increment_angles(self, action, time_step_size):
        """
        Increment the angles of the joints.
        :param action: action calculated using equation 3 in report.
        :param time_step_size: float, dt value.
        """
        self.joint_angle += self._calc_diff_angle(action, time_step_size)

    def draw(self, dashed: bool = False):
        """
        Draws the arm into a figure. If a figure is already active, it takes
        that one.
        :param dashed: boolean, option to make the arm dashed.
        """
        plt.rcParams.update({'font.size': 20})
        if not plt.get_fignums():
            size = 10
            fig = plt.figure(figsize=(size, size / 1.5))
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

        x_axes_range = np.array([-0.5 / 3, 2.5 / 3]) * self.n_joints
        y_axes_range = np.array([-1.5 / 3, 1 / 3]) * self.n_joints

        if self.n_joints == 3:
            x_ticks = np.linspace(x_axes_range[0], x_axes_range[1], 7)
            y_ticks = np.linspace(y_axes_range[0], y_axes_range[1], 6)
        elif self.n_joints == 100:
            x_ticks = np.linspace(0, 100, 6)
            y_ticks = np.linspace(-30, 30, 7)
            y_axes_range = np.array([-35, 35])
        else:
            x_ticks = np.linspace(x_axes_range[0], x_axes_range[1], 7)
            y_ticks = np.linspace(y_axes_range[0], y_axes_range[1], 7)

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xlim(x_axes_range)
        ax.set_ylim(y_axes_range)
        if dashed:
            ax.plot(x_joints, y_joints, marker='o', ls=':',
                    mec='orange', mfc='orange', c='blue')
        else:
            ax.plot(x_joints, y_joints, marker='o',
                    mec='red', mfc='red', c='blue')
