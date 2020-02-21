import numpy as np
from scipy.optimize import fsolve, newton, root
from functools import partial
from week_9.arm import Arm
import matplotlib.pyplot as plt
import copy


def get_functions(arm, time_window, time, target, initial_state, alpha=0.1):
    # separating the x and y from target, and mu and sigma from init
    x_target, y_target = target
    mu, sigma = initial_state[:arm.n_joints], initial_state[arm.n_joints:]

    # intialisation of array containing the equations
    eqns = np.zeros(2 * arm.n_joints)

    # calculating expected x and y for all joints
    x_expected = np.sum(np.cos(mu) * np.exp(- (sigma ** 2) / 2))
    y_expected = np.sum(np.sin(mu) * np.exp(- (sigma ** 2) / 2))
    for i in range(arm.n_joints):
        # mu equations
        eqns[i] = \
            mu[i] + \
            alpha * (time_window - time) * (
                    np.sin(mu[i]) * np.exp(- (sigma[i] ** 2) / 2) *
                    (x_expected - x_target) -
                    np.cos(mu[i]) * np.exp(- (sigma[i] ** 2) / 2) *
                    (y_expected - y_target)
            )
        # sigma equations
        eqns[arm.n_joints + i] = 1 / np.sqrt(
                1 / arm.noise_parameter * (
                        1 / (time_window - time) +
                        alpha * np.exp(- sigma[i] ** 2) -
                        alpha * (x_expected - x_target) *
                        np.cos(mu[i]) * np.exp(- (sigma[i] ** 2) / 2) -
                        alpha * (y_expected - y_target) *
                        np.sin(mu[i]) * np.exp(- (sigma[i] ** 2) / 2)
                )
            )
    return eqns


def solve(function, x_init, error_range: float = 0.01):
    """
    Solves the given function using fixed point iteration.
    :param function: function to solve, should only take x_init as parameter.
    :param x_init: initial point to start the solving method from.
    :param error_range: error difference between x_new - x_old for accepting
    x_new
    :return: ndarray of shape (x_init,).
    """
    error = 1
    x_old = x_init
    while error > error_range:
        x_new = function(x_old)
        error = np.abs(x_new - x_old).sum()
        x_old = x_new
    return x_new


def control(max_time: float, current_time: float,
            expected_angle, initial_angle):
    """
    Control function used to calculate the amount of action to give to the arm
    for it to move
    :param max_time:
    :param current_time:
    :param expected_angle:
    :param initial_angle:
    :return:
    """
    return (1 / (max_time - current_time)) * (expected_angle - initial_angle)


def move_arm(arm: Arm, move_to, max_time: float, n_steps: int):
    time_step = max_time / n_steps
    times = np.arange(max_time, step=time_step)
    init = np.random.normal(size=arm.n_joints * 2, scale=10)
    for t in times:
        to_solve = partial(get_functions, arm, max_time, t, move_to)
        solution = solve(to_solve, init)
        expected_angles = solution[:arm.n_joints]
        sigma = solution[arm.n_joints:]
        action = control(max_time, t, expected_angles, arm.joint_angle)
        arm.increment_angles(action, time_step)

        expected_arm = copy.deepcopy(arm)
        expected_arm.joint_angle = expected_angles
        expected_arm.draw(dashed=True)
        arm.draw()
        plt.show()

        init = np.concatenate((arm.joint_angle, sigma))

