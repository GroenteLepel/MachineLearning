import numpy as np
from functools import partial
from week_9.arm import Arm
import matplotlib.pyplot as plt
import copy
import progress_bar.progress_bar as pb

def get_functions(arm, time_window, time, target, initial_state, alpha=0.1):
    """
    Function for creating the mu and sigma equations describing the means and
    standard deviations for the theta_i of the arm.
    :param arm: Arm object for which the mu's and sigma's must be determined.
    :param time_window: float, max time delivered for the arm to move.
    :param time: float, current time of the iteration.
    :param target: 2d array coordinate indicating the target to move to.
    :param initial_state: current mu and sigma values of the arm.
    :param alpha: float, factor for taking the influence of the mu and
    sigma calculation, default is 0.1
    :return: ndarray of shape (2 * arm.n_joints,) containing the mu's first,
    followed by the sigmas.
    """
    # separating the x and y from target, and mu and sigma from init
    x_target, y_target = target
    mu, sigma2 = initial_state[:arm.n_joints], initial_state[arm.n_joints:]

    # intialisation of array containing the equations
    eqns = np.zeros(2 * arm.n_joints)

    # calculating expected x and y for all joints
    x_expected = np.sum(np.cos(mu) * np.exp(- (sigma2) / 2))
    y_expected = np.sum(np.sin(mu) * np.exp(- (sigma2) / 2))
    for i in range(arm.n_joints):
        # mu equations
        eqns[i] = \
            (mu[i] + \
            alpha * (time_window - time) * (
                    np.sin(mu[i]) * np.exp(- (sigma2[i]) / 2) *
                    (x_expected - x_target) -
                    np.cos(mu[i]) * np.exp(- (sigma2[i]) / 2) *
                    (y_expected - y_target)
            ))
        # sigma equations
        eqns[arm.n_joints + i] =  1 / ( 1 / arm.noise_parameter * (
                        1 / (time_window - time) +
                        alpha * np.exp(- sigma2[i]) -
                        alpha * (x_expected - x_target) *
                        np.cos(mu[i]) * np.exp(- (sigma2[i]) / 2) -
                        alpha * (y_expected - y_target) *
                        np.sin(mu[i]) * np.exp(- (sigma2[i]) / 2)
                ))
    return eqns


def solve(function, x_init, arm, error_range: float = 0.001):
    """
    Solves the given function using fixed point iteration.
    :param function: function to solve, should only take x_init as parameter.
    :param x_init: initial point to start the solving method from.
    :param error_range: error difference between x_new - x_old for accepting
    x_new
    :return: ndarray of shape (x_init,).
    """
    # scale the error range with n_joints
    error_range = error_range * (arm.n_joints / 3) ** 1.2

    # initialise the values for solving
    error, x_old, eta = 1, x_init, 0.01 / 3
    x_new = np.zeros(2 * arm.n_joints)

    # smoothed fixed point iteration
    while error > error_range:
        x_new = x_old * (1 - eta) + eta * function(x_old)
        error = np.abs(x_new - function(x_new)).sum()
        x_old = x_new

    return x_new


def control(max_time: float, current_time: float,
            expected_angle, initial_angle):
    """
    Control function used to calculate the amount of action to give to the arm
    for it to move.
    :param max_time: float, max time delivered for the arm to move.
    :param current_time: current time step.
    :param expected_angle: final expected angle drawn from the probability
    distribution (equation 4 in report).
    :param initial_angle: current or initial angle of the arm.
    :return: ndarray, dtype=float
    """
    return (1 / (max_time - current_time)) * (expected_angle - initial_angle)


def move_arm(arm: Arm, move_to, max_time: float, n_steps: int):
    """
    Moves the arm to the desired coordinate move_to within time max_time using
    n_steps as number of steps.
    :param arm: Arm object which has to move to desired coordinate.
    :param move_to: 2d array, dtype=float, indicating the final coordinate.
    :param max_time: float, max time delivered for the arm to move.
    :param n_steps: int, number of steps the arm can take to move within
    max_time
    """
    # create array for all time iterations
    times = np.linspace(0, max_time, n_steps)
    times[-1] -= 1e-3  # to prevent nan values in final iteration
    time_step = times[1] - times[0]

    # initialise the init array for fixed point iteration, chosing random values
    init = np.random.normal(size=arm.n_joints * 2)
    for t in times:
        # printing bar for verbosity
        percentage = t / max_time * 100
        bar = pb.percentage_to_bar(percentage)
        print('\r', bar, end='')

        # creating the equations to solve
        to_solve = partial(get_functions, arm, max_time, t, move_to)
        # solve the equations using fixed point iteration
        solution = solve(to_solve, init, arm)

        # derive the individual solutions
        expected_angles = solution[:arm.n_joints]
        sigma2 = solution[arm.n_joints:]
        # sigma = np.sqrt(1 / solution[arm.n_joints:])

        # calculate action and increment the angles in the arm
        action = control(max_time, t, expected_angles, arm.joint_angle)
        arm.increment_angles(action, time_step)

        # draw the expectation value of the arm for reference
        expected_arm = copy.deepcopy(arm)
        expected_arm.joint_angle = expected_angles
        expected_arm.draw(dashed=True)
        # draw the current arm
        arm.draw()

        # plot
        plt.title("t = {0:.2f}".format(t))
        filename = "t{0:.2f}".format(t).replace('.', '')
        plt.savefig("../data/arm_move/{}".format(filename))
        plt.clf()

        # change the init to the current values for kickstarting the fixed
        #  point iteration
        init = np.concatenate((arm.joint_angle, sigma2))

