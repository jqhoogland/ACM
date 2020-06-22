# ------------------------------------------------------------

# Code to run experiments and run simulations on the KCMs defined in kcm.py

# ------------------------------------------------------------

import matplotlib.pyplot as pyplot
import numpy as np
import tables
import os

from fa_discrete import *

def run(eps, n_sites, t_obs, s_):
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
    filepath = "{}/eps={}_nSites={}_tObs={}.h5".format(save_dir, eps, n_sites, t_obs)     

    save_file = tables.open_file(filepath, mode="w")
    atom = tables.Float64Atom()

    save_array = save_file.create_earray(save_file.root, 'data', atom,(0, 3))
    save_file.close()

    for s in s_:
        sfa = SoftenedFA(s=s, eps=eps, gamma=0.25, num_sites=n_sites, num_steps=t_obs, num_burnin_steps=100)
        trajectory = sfa.gen_trajectory(draw=False)
        sfa_tps = SoftenedFATPS(sfa, sfa.activity)
        k_avg, k_err = sfa_tps.mc_analysis(100)

        save_obj = np.array([[s, k_avg, k_err]])

        save_file = tables.open_file(filepath, mode="a")
        save_file.root.data.append(save_obj)
        save_file.close()


    msg = "All results saved to:{}".format(filepath)
    return msg


if __name__ == "__main__":

    s_ = np.array([-1, 0, 1])
    msg = run(eps=0, n_sites=10, t_obs=10, s_=s_)

    filepath = msg.split(":")[1]
    load_file = tables.open_file(filepath, mode="r")
    print(load_file.root.data[:, :])
    load_file.close()