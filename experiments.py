# ------------------------------------------------------------

# Code to run experiments and run simulations on the KCMs defined in kcm.py

# ------------------------------------------------------------

import matplotlib.pyplot as pyplot
import numpy as np
import tables
import os

from fa_discrete import *
from draw import *


def run_fixed_eps(run_id, eps, n_sites, t_obs, s_):
    """
    Run an experiment for given parameters and save the results

    :param eps float: fixed value for epsilon for the sFA model
    :param n_sites int: number of sites of the trajectories
    :param t_obs int: number of time steps of the trajectories
    :param s_ np.array: array of s field values for the sFA model

    :return msg string: return message indicating the succes of the function
    """
    save_dir = "results/"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    filepath = "{}/runID={}_eps={}_nSites={}_tObs={}.h5".format(save_dir, run_id, eps, n_sites, t_obs)     

    save_file = tables.open_file(filepath, mode="w")
    atom = tables.Float64Atom()

    save_array = save_file.create_earray(save_file.root, 'data', atom,(0, 3))
    save_array.append(np.array([[eps, n_sites, t_obs]]))
    save_file.close()

    for s in s_:
        sfa = SoftenedFA(s=s, eps=eps, gamma=0.25, num_sites=n_sites, num_steps=t_obs, num_burnin_steps=100)
        sfa_tps = SoftenedFATPS(sfa, sfa.activity)
        k_avg, k_err = sfa_tps.mc_analysis(1000, 100, draw=False)

        save_obj = np.array([[s, k_avg, k_err]])

        save_file = tables.open_file(filepath, mode="a")
        save_file.root.data.append(save_obj)
        save_file.close()


    msg = "All results saved to:{}".format(filepath)
    return msg


def create_s_vals(s_critical, n_s_steps):
    s_ = [s_critical]
    for i in reversed(range(int(n_s_steps/2) + 1)):
        s_.insert(0, s_critical - 0.5*10**-i)
        s_.append(s_critical + 0.5*10**-i)
        s_.insert(0, s_critical - 10**-i)
        s_.append(s_critical + 10**-i)

    return s_


if __name__ == "__main__":

    s_ = create_s_vals(0, 6)
    print(s_)
    msg = run_fixed_eps(run_id=1, eps=0, n_sites=60, t_obs=1000, s_=s_)

    filepath = msg.split(":")[1]
    load_file = tables.open_file(filepath, mode="r")
    eps, n_sites, t_obs = load_file.root.data[0, :]
    data = load_file.root.data[1:, :]
    load_file.close()

    s_vals, k_avgs, k_errs = data.transpose()
    draw_activity_for_fixed_eps(eps, s_vals, k_avgs, k_errs)
    plt.show()