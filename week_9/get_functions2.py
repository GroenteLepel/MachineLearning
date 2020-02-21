import numpy as np

def get_functions2(arm, time_window, time, target, alpha):
    'I wrote a slightly different set-up'

    def f(z):
        x_target, y_target = target
        nu = (arm.noise_parameter)
        mu, sigma = z[:arm.n_joints], z[arm.n_joints:]
        a = np.zeros(2 * arm.n_joints)
        x = np.sum(np.cos(mu) * np.exp(- (sigma ** 2) / 2))
        y = np.sum(np.sin(mu) * np.exp(- (sigma ** 2) / 2))
        for i in range(arm.n_joints):
            a[i] = \
                (0 + (
                        arm.joint_angle[i] +
                        alpha * (time_window - time) * (
                                np.sin(mu[i]) * np.exp(- (sigma[i]) ** 2 / 2) *
                                (x - x_target) -
                                np.cos(mu[i]) * np.exp(- (sigma[i] ** 2) / 2) *
                                (y - y_target)
                        )))
            a[arm.n_joints + i] = \
                np.abs(np.sqrt(1 / (
                        1. / nu * (
                        1. / (time_window - time) +
                        alpha * np.exp(- sigma[i] ** 2) -
                        alpha * (x - x_target) *
                        np.cos(mu[i]) * np.exp(- sigma[i] ** 2 / 2) -
                        alpha * (y - y_target) *
                        np.sin(mu[i]) * np.exp(- sigma[i] ** 2 / 2)
                ))))
        return a

    return f