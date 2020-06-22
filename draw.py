import matplotlib.pyplot as plt


def draw_average_plot(measurements, title, xlabel, ylabel):
    num_measurements = len(measurements)

    plt.figure(figsize=(20,3))
    plt.plot(np.arange(num_measurements), measurements)
    plt.plot(np.arange(num_measurements), np.ones(num_measurements) * np.mean(measurements))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(top=1., bottom=0.)
    plt.show()

def draw_activity_for_fixed_eps(eps, s_vals, activity_means, activity_errs=None):
    """

    :param (float) eps: The value of epsilon (used for labelling the curve).
    :param (np.array of shape [num_s_vals, ]) s_vals: The values of the s field
        for which observables have been measured
    :param (np.array of shape [num_s_vals, ]) activity_means: The average activity
        corresponding to the values of s in ``s_vals``
    :param (Optional np.array of shape [num_s_vals, ]) activity_errors: The error
        of activity corresponding to the values of s in ``s_vals``. If not provided,
        defaults to None, and the error bars are not included in the plot.
    """

    plt.errorbar(s_vals, activity_means, label="$\epsilon={}$".format(eps), yerr=activity_errs)
    plt.title("Activity as a function of $s$ for fixed $\epsilon$")
    plt.xlabel("Dynamical Field, $s$")
    plt.ylabel("Activity, $K$")
    plt.legend()
