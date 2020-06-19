# ------------------------------------------------------------

# Code to run experiments and run simulations on the KCMs defined in kcm.py

# ------------------------------------------------------------

import matplotlib.pyplot as pyplot
import numpy as np

from kcm import *

def sfa_parameter_search(eps_, s_, draw=False):
    """
    Perform a grid search for the paramers `eps` and `s` and visualize the results.

    :param list eps_: list of values for eps that will be considered
    :param list s_: list of values for s that will be considered
    :param bool draw: If True will draw the trajectories that have been generated. The trajectories are shown and not saved.
    """

    # check if input is correct
    assert len(eps_) > 0
    assert len(s_) > 0

    activations = np.zeros((len(eps_), len(s_)))

    for i, eps in enumerate(eps_):
        for j, s in enumerate(s_):
            print("Softening parameter, epsilon: {}; Biasing field, s: {}".format(eps, s))
            fa_kcm = SoftenedFA(gamma=0.25, s=-s, eps=eps, num_burnin_steps=0, num_sites=60, num_steps=600)
            trajectory = fa_kcm.gen_trajectory()
            tps = TransitionPathSampler(fa_kcm, fa_kcm.activity)
            tps.mc_average(100)
            activations[i,j] = fa_kcm.activity(trajectory)

            if draw:
                print("Activity: {}".format(fa_kcm.activity(trajectory)))
                draw_trajectory(trajectory)

    plt.figure()
    for i in range(len(eps_)):
        print(activations[i, :])
        plt.scatter(x=s_, y=activations[i, :], label="eps = {}".format(eps_[i]))
    plt.legend()
    plt.ylim(0,1)
    plt.xlabel("s")
    plt.ylabel("k(s)")
    plt.show()

def east_parameter_search(s_):

    assert len(s_) > 0

    activations = []

    for s in s_:
        east_kcm = EastKCM(prob_transition=s, num_burnin_steps=0, num_sites=60, num_steps=600)
        trajectory = east_kcm.gen_trajectory()
        tps = TransitionPathSampler(east_kcm, east_kcm.activity)
        activations.append(tps.mc_average(100))

    plt.figure()
    for i in range(len(s_)):
        plt.scatter(x=s_, y=activations)
    plt.xlabel("s")
    plt.ylabel("average activation of 100 samples")
    plt.show()

def grid_search_one_spin_fa(steps):
    flip_probs = [i/float(steps) for i in range(steps)]
    swap_probs = [i/float(steps) for i in range(steps)]

    activities = np.zeros((steps, steps))

    for i, flip_prob in tqdm(enumerate(flip_probs)):
        for j, swap_prob in enumerate(swap_probs):
            fa_model = OneSpinFAKCM(flip_prob, swap_prob, 60, 600, num_burnin_steps=0)
            trajectory = fa_model.gen_trajectory()
            activities[i, j] = fa_model.activity(trajectory)
    
    plt.figure()
    plt.imshow(activities, extent=[0, 1, 0, 1])
    plt.colorbar()
    plt.xlabel("swap probability")
    plt.ylabel("flip probability")
    plt.title("Activity of OneSpinFAKCM for multiple parameter settings")
    plt.show()
            



if __name__ == "__main__":

    grid_search_one_spin_fa(steps=10)