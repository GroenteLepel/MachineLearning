import numpy as np
from scipy.optimize import fsolve, newton, root
from functools import partial
#from week_9.arm import Arm
#from arm.py import Arm


def get_functions(arm, time_window, time, target, alpha, p):
    x_target, y_target = target
    nu = arm.noise_parameter
    mu, sigma = p[:arm.n_joints], p[arm.n_joints:]
    a = np.zeros(2 * arm.n_joints)
    x = np.sum(np.cos(mu) * np.exp(- (sigma ** 2) / 2))
    y = np.sum(np.sin(mu) * np.exp(- (sigma ** 2) / 2))
    for i in range(arm.n_joints):
        a[i] = \
            - mu[i] + \
            arm.joint_angle[i] + \
            alpha * (time_window - time) * (
                    np.sin(mu[i]) * np.exp(- (sigma[i] ** 2) / 2) *
                    (x - x_target) -
                    np.cos(mu[i]) * np.exp(- (sigma[i] ** 2) / 2) *
                    (y - y_target)
            )
        a[arm.n_joints + i] = \
            - 1 / (sigma[i] ** 2) + \
            1 / nu * (
                    1 / (time_window - time) +
                    alpha * np.exp(- sigma[i] ** 2) -
                    alpha * (x - x_target) *
                    np.cos(mu[i]) * np.exp(- (sigma[i] ** 2) / 2) -
                    alpha * (y - y_target) *
                    np.sin(mu[i]) * np.exp(- (sigma[i] ** 2) / 2)
            )
    return a

def get_functions2(arm, time_window, time, target, alpha):
    
    'I wrote a slightly different set-up' 
    
    def f(z):
        x_target, y_target = target
        nu = (arm.noise_parameter)
        mu, sigma2 = z[:arm.n_joints] , z[arm.n_joints:]
        a = np.zeros(2 * arm.n_joints)
        x = np.sum(np.cos(mu) * np.exp(- (sigma2) / 2))
        y = np.sum(np.sin(mu) * np.exp(- (sigma2) / 2))
        for i in range(arm.n_joints):
            a[i] = \
                 (- mu[i] +( \
                arm.joint_angle[i] + \
                alpha * (time_window - time) * (
                        np.sin(mu[i]) * np.exp(- (sigma2[i]) / 2) *
                        (x - x_target) -
                        np.cos(mu[i]) * np.exp(- (sigma2[i]) / 2) *
                        (y - y_target)
                ))) 
            a[arm.n_joints + i] = \
                 -(1. / sigma2[i]) + ( \
                1. / nu * (
                        1. / (time_window - time) +
                        alpha * np.exp(- sigma2[i]) -
                        alpha * (x - x_target) *
                        np.cos(mu[i]) * np.exp(- sigma2[i] / 2) -
                        alpha * (y - y_target) *
                        np.sin(mu[i]) * np.exp(- sigma2[i] / 2)
                ))
        return a
    return f


def control(max_time: float, current_time: float,
            expected_angle, initial_angle):
    return (1 / (max_time - current_time)) * (expected_angle - initial_angle)


#def angle_increment(control, time_step_size, noise):
#    return control * time_step_size + noise


def calculate_step():
    max_time = 1
    n_steps = 100
    n_plots = 10
    time_step = max_time / n_steps
    times = np.arange(max_time, step=time_step)
    my_arm = Arm(3, 0.01, angles=[0.01, 0.02, 0.01])
    for t in times:
       # to_solve = partial(get_functions, my_arm, max_time, t, [0.0, 0.0], 0.1)
        solution = np.concatenate(([-1,-1,-1], [1,1,1]))
        while np.abs(np.sum(np.cos(solution[:my_arm.n_joints]))) + \
                     np.abs(np.sum(np.cos(solution[:my_arm.n_joints]))) > 0.01:
                     to_solve = get_functions2(my_arm, max_time, t, [0.0, 0.0], 0.1) 
                     start = np.concatenate(([0.1,0,3.14/0.75], [100,100,100]))
                     solution = fsolve(to_solve,\
                                np.concatenate((np.random.normal(size = 3, loc = 0, scale = 2) \
                                               , [1,1,1])))
        action = control(max_time, t,
                         solution[:my_arm.n_joints], my_arm.joint_angle)
        my_arm.increment_angles(action, time_step)
        
   #     if (t / time_step) % (1 / n_plots) == 0:
    #        my_arm.show()
     #   my_arm.show()
     
    return my_arm


# # my_arm = Arm(3, 0.1, angles=[np.pi, 0.5 * np.pi, np.pi])
# my_arm = Arm(3, 0.1, angles=[0, 0, 0])
# my_arm.show()

#
solved_arm = calculate_step()
print(solved_arm.joint_angle)
solved_arm.show()

my_arm = Arm(3, 0.1, angles=[0, 0.0, 0.])
test = get_functions2(my_arm, 3, 0, [0.0, 0.0], 1)
np.random.seed(3)
root(test, np.concatenate((np.random.normal(size =3), [1,1,1])))['x']
np.random.seed(3)
fsolve(test, np.concatenate(([0,0,0], [1,1,1])))
def g(x):
    return x ** 2 - 3

fsolve(g, 0)

test(np.concatenate((np.random.normal(size =3), [1,1,1])))
