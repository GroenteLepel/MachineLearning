import numpy as np
from scipy.optimize import fsolve, newton, root
from functools import partial
from week_9.arm import Arm
import matplotlib.pyplot as plt
import copy


def get_functions(arm, time_window, time, target, initial_state, alpha=0.1):
    x_target, y_target = target
    mu, sigma = initial_state[:arm.n_joints], initial_state[arm.n_joints:]
    eqns = np.zeros(2 * arm.n_joints)  # array containing the equations
    x_expected = np.sum(np.cos(mu) * np.exp(- (sigma ** 2) / 2))
    y_expected = np.sum(np.sin(mu) * np.exp(- (sigma ** 2) / 2))
    for i in range(arm.n_joints):
        eqns[i] = \
            arm.joint_angle[i] + \
            alpha * (time_window - time) * (
                    np.sin(mu[i]) * np.exp(- (sigma[i] ** 2) / 2) *
                    (x_expected - x_target) -
                    np.cos(mu[i]) * np.exp(- (sigma[i] ** 2) / 2) *
                    (y_expected - y_target)
            )
        eqns[arm.n_joints + i] = \
            1 / arm.noise_parameter * (
                    1 / (time_window - time) +
                    alpha * np.exp(- sigma[i] ** 2) -
                    alpha * (x_expected - x_target) *
                    np.cos(mu[i]) * np.exp(- (sigma[i] ** 2) / 2) -
                    alpha * (y_expected - y_target) *
                    np.sin(mu[i]) * np.exp(- (sigma[i] ** 2) / 2)
            )
    return eqns


def solve(function, x_init, error_range: float = 0.01):
    error = 1
    x_old = x_init
    while error > error_range:
        x_new = function(x_old)
        error = (x_new - x_old).sum()
        x_old = x_new
    return x_new


def control(max_time: float, current_time: float,
            expected_angle, initial_angle):
    return (1 / (max_time - current_time)) * (expected_angle - initial_angle)


def move_arm(arm: Arm, move_to, max_time: float, n_steps: int):
    time_step = max_time / n_steps
    times = np.arange(max_time, step=time_step)
    init = np.random.normal(size=arm.n_joints * 2)
    for t in times:
        expected_arm = copy.deepcopy(arm)
        to_solve = partial(get_functions, arm, max_time, t, move_to)
        solution = solve(to_solve, init)
        action = control(max_time, t, solution[:arm.n_joints], arm.joint_angle)

        expected_arm.joint_angle = solution[:expected_arm.n_joints]
        arm.increment_angles(action, time_step)
        expected_arm.draw(dashed=True)
        arm.draw()
        plt.show()

        init = np.concatenate((arm.joint_angle, solution[arm.n_joints:]))

