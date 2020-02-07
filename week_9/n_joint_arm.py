import numpy as np
from scipy.optimize import fsolve
from functools import partial
from week_9.arm import Arm


def get_functions(arm, time_window, time, target, alpha, p):
    x_target, y_target = target
    nu = arm.noise
    mu, sigma = p[:arm.n_joints], p[arm.n_joints:]
    a = np.zeros(2 * arm.n_joints)
    x = np.sum(np.cos(mu) * np.exp(- (sigma ** 2) / 2))
    y = np.sum(np.sin(mu) * np.exp(- (sigma ** 2) / 2))
    for i in range(arm.n_joints):
        a[i] = \
            - mu[i] + arm.joint_angle[i] + alpha * (time_window - time) * (
                    np.sin(mu[i]) * np.exp(- (sigma[i] ** 2) / 2) * (
                    x - x_target) - np.cos(
                mu[i] * np.exp(- (sigma[i] ** 2) / 2) * (y - y_target)))
        a[arm.n_joints + i] = \
            - 1 / (sigma[i] ** 2) + 1 / nu * (
                    1 / (time_window - time) + alpha * np.exp(- sigma[i] ** 2) -
                    alpha * (x - x_target) * np.cos(mu[i]) * np.exp(
                - (sigma[i] ** 2) / 2) -
                    alpha * (y - y_target) * np.sin(mu[i]) * np.exp(
                - (sigma[i] ** 2) / 2)
            )
    return a


def calculate_step():
    max_time = 2
    time_step = 0.1
    times = np.arange(max_time, step=time_step)
    my_arm = Arm(3, 0.1)
    for t in times:
        to_solve = partial(get_functions, my_arm, 2, t, [0, 0], 2)
        start = np.ones(2 * my_arm.n_joints)
        solution = fsolve(to_solve, start)
        action = 1 / (max_time - t) * (
                    solution[:my_arm.n_joints] - my_arm.joint_angle)
        my_arm.joint_angle = action * time_step + np.sqrt(
            my_arm.noise * time_step) * np.random.normal(size=my_arm.n_joints)


my_arm = Arm(3, 0.1, angles=[np.pi, 0.5 * np.pi, np.pi])
# my_arm = Arm(3, 0.1, angles=[0, 0, 0])
my_arm.show()