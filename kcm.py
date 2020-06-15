# ------------------------------------------------------------

# Code to implement Kinetically Constrained Models
#
# *References*
# [1] The Fredrickson-Aldersen Model [Frederickson and Alderson 1984](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.53.1244)
# [2] The East Model [Jackle and Eisinger 1991](https://link.springer.com/article/10.1007/BF01453764)
# [3] [Nicholas B. Tutto 2018](https://tps.phys.tue.nl/lm-live/fa-ising/)

# ------------------------------------------------------------

import logging
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize

class KCM(object):
    """

    For convenience sake, we interpret the spins as occupation numbers $$0, 1$$
    rather than the conventional up/down $$\pm 1$$.

    This class produces a single trajectory.

    """

    def __init__(self, prob_transition, num_sites, num_steps, num_burnin_steps=0):
        """
        :param float trans_prob: The probability (with value in range [0, 1])
            of a spin flipping if its right neighbor is up (i.e. =1).
        :param int num_sites: The number of sites to include in the chain.
        :param int num_steps: The number of steps to include in a trajectory.
        :param int num_burnin_steps: The number of steps to equilibrate a
            randomly initialized configuration.

        """
        assert prob_transition <= 1 and prob_transition >= 0
        assert num_sites > 1

        self.prob_transition = prob_transition
        self.num_sites = num_sites
        self.num_steps = num_steps
        self.num_burnin_steps = num_burnin_steps

    def _step(self, state):
        raise NotImplementedError("")

    def gen_trajectory(self, init_state=None):
        """
        Generates a trajectory of ``self.num_steps`` steps, preceded by
        ``self.num_burnin_step`` burn-in steps.

        :param (np.array or None) init_state: An optional numpy array of 0s and
            1s that should have size ``(self.num_sites, )``. If None, defaults
            to a randomly chosen array of 0s and 1s.


        :returns (np.array) trajectory: A trajectory of shape
            (self.num_steps, self.num_sites). The rows are states.

        Note: if ``self.num_burnin_steps > 0``, then ``init_state`` will NOT be
            the first entry in trajectories
        """

        state = init_state
        trajectory = np.zeros((self.num_steps, self.num_sites))

        if init_state is None:
            state = np.random.choice([0, 1], size=(self.num_sites, ))

        assert state.shape == (self.num_sites, )

        for _ in range(self.num_burnin_steps - 1): # -1 because we update state once more in the forloop below.
            state = self._step(state)

        for i in range(self.num_steps):
            state = self._step(state)
            trajectory[i, :] = state

        return trajectory

    def activity(self, trajectory):
        activity = 0.
        for i in range(self.num_steps - 1):
            activity += np.sum(trajectory[i, :] != trajectory[i + 1, :])

        return activity /(self.num_steps * self.num_sites)



class EastKCM(KCM):
    """

    The East Model is a kinetic 1-dimensional Ising chain.
     - It's spins are non-interacting.
     - Spin ``i`` can flip with probability ``prob_transition`` iff spin ``i+1``
       (to the right, i.e. "east") is "up"
     - Boundaries are closed.

    For convenience sake, we interpret the spins as occupation numbers $$0, 1$$
    rather than the conventional up/down $$\pm 1$$.

    This class produces a single trajectory.

    """

    def __init__(self, prob_transition, num_sites, num_steps, num_burnin_steps=0):
        """
        :param float trans_prob: The probability (with value in range [0, 1])
            of a spin flipping if its right neighbor is up (i.e. =1).
        :param int num_sites: The number of sites to include in the chain.
        :param int num_steps: The number of steps to include in a trajectory.
        :param int num_burnin_steps: The number of steps to equilibrate a
            randomly initialized configuration.

        """
        super(EastKCM, self).__init__(prob_transition, num_sites, num_steps, num_burnin_steps)

    def _step(self, state):
        """
        Carries out one step on ``state``.
        Iterate over all indices and probabilistically update (with probability
        ``self.prob_transition``) those spins whose right-neighbors have value 0

        """

        dummie = state[0]

        for i in range(self.num_sites - 1):
            if state[i + 1] == 1 and np.random.uniform() < self.prob_transition:
                state[i] = 1 - state[i]

        if dummie == 1 and np.random.uniform() < self.prob_transition:
            state[-1] = 1 - state[-1]

        return state

class TransitionPathSampler(object):
    """

    Described in Supporting Information of Elmatad et al. 2010

    """
    def __init__(self, kcm, observable):
        """
        Takes a kcm model, and can measure the observable over ensembles of trajectories taken from the kcm
        """

        self.kcm = kcm
        self.num_steps = kcm.num_steps
        self.observable = observable

    def _half_shoot(self, trajectory):
        i = np.random.choice(np.arange(self.kcm.num_sites, dtype=np.int32))

        for j in range(i, self.kcm.num_steps -1):
            trajectory[j + 1, :] = self.kcm._step(trajectory[j, :])

        return trajectory

    def _shift(self, trajectory):
        i = np.random.choice(np.arange(self.kcm.num_sites, dtype=np.int32))

        trajectory[:self.kcm.num_steps - i, :] = trajectory[i:, :]

        for j in range(self.kcm.num_steps - i, self.kcm.num_steps -1):
            trajectory[j + 1, :] = self.kcm._step(trajectory[j, :])

        return trajectory

    def _step(self, trajectory):
        # With equal probability we either half-shoot or shift.
        if np.random.uniform() < 0.5:
            return self._half_shoot(trajectory)
        else:
            return self._shift(trajectory)

    def get_trajectory_weight(self, trial_trajectory):
        # See ``accept_trajectory``
        # By default, we except all newly generated trajectories.
        return 1.

    @staticmethod
    def accept_trajectory(energy_prev, energy_curr):
        # By default, we except all newly generated trajectories.
        return True

    def mc_samples(self, num_samples, num_burnin_in_steps=0, verbose=False):
        # TODO: also measure the standard deviations (or do a fancier binning analysis)!

        measurements = np.zeros(num_samples)

        trajectory = self.kcm.gen_trajectory()

        energy_prev = self.get_trajectory_weight(trajectory)
        energy_curr = energy_prev

        for i in range(num_samples):
            measurements[i] = self.observable(trajectory)

            if verbose:
                draw_trajectory(trajectory)

            trial_trajectory = self._step(trajectory)

            energy_prev = energy_curr
            energy_curr = self.get_trajectory_weight(trial_trajectory)

            if self.accept_trajectory(energy_prev, energy_curr):
                trajectory = trial_trajectory

        return measurements

    def mc_average(self, num_samples, num_burnin_in_steps=0, verbose=False):
        return np.mean(self.mc_samples(num_samples, num_burnin_in_steps, verbose))

class SoftenedFATPS(TransitionPathSampler):
    def __init__(self, *args):
        """
        docstring
        """
        super(SoftenedFATPS, self).__init__(*args)
        self.alpha = np.arctan(2. * np.exp(-self.kcm.s) * (1. + self.kcm.eps) / (2. +self.kcm.eps - self.kcm.eps * self.kcm.gamma))
        self.g = np.log(np.tan(self.alpha / 2))


    def get_trajectory_weight(self, trajectory):
        return self.kcm.s * self.kcm.activity(trajectory) - self.g * (np.sum(trajectory[0, :] + trajectory[-1, :]))

    @staticmethod
    def accept_trajectory(energy_prev, energy_curr):
        return np.random.uniform() < min(1, np.exp(energy_prev - energy_curr))



class OneSpinFAKCM(KCM):
    """

    The one-spin Fredrickson-Andersen Model is a kinetic 1-dimensional Ising chain.
     - It's spins are non-interacting.
     - Spin ``i`` can flip with probability ``prob_transition`` iff spin ``i+1``
       (to the right or left, i.e. "east") is "up"
     - Boundaries are closed.

    For convenience sake, we interpret the spins as occupation numbers $$0, 1$$
    rather than the conventional up/down $$\pm 1$$.

    This class produces a single trajectory.

    """

    def __init__(self, prob_transition, prob_swap, num_sites, num_steps, coupling_energy, control_parameter=0., neighbor_constraint=1, num_burnin_steps=0):
        """
        :param float prob_transition: The probability (with value in range [0, 1])
            of a spin flipping if its right neighbor is up (i.e. =1).
        :param int num_sites: The number of sites to include in the chain.
        :param int num_steps: The number of steps to include in a trajectory.
        :param int num_burnin_steps: The number of steps to equilibrate a
            randomly initialized configuration.

        """
        super(OneSpinFAKCM, self).__init__(prob_transition, num_sites, num_steps, num_burnin_steps)
        self.external_field = control_parameter
        self.coupling_energy = coupling_energy
        self.prob_swap = prob_swap
        self.neighbor_constraint = neighbor_constraint

    def _step(self, state):
        """
        Carries out one step on ``state``.
        Iterate over all indices and probabilistically update (with probability
        ``self.prob_transition`` times the number of neighboring up spins)
        those spins whose neighbors have value 0

        """

        # define a random order by which to go through the state
        shuffled_indices = np.random.permutation(self.num_sites)

        tmp_state = state.copy()

        for index in shuffled_indices:

            # check if the neighbor contraint is upheld
            if not self._check_neighbor_constraint(state, index):
                continue

            # attempt a flip
            if np.random.uniform() > self.prob_transition:

                # calculate the change in energy of system after attempted move
                dE = self._energy_change_of_flip(state, index)

                # acceptence probability of the move depends on the change in energy
                if np.random.uniform() < min(1., np.exp(-dE)):
                    tmp_state[index] = 1 - state[index]

            # attempt a swap
            if np.random.uniform() > self.prob_swap and index < len(state) - 1:
                direction = np.random.choice([-1, 1])

                # calculate the change in energy of the system after a swap
                dE = self._energy_change_of_swap(state, index, direction)

                # acceptence probability of the move depends on the change in energy
                if np.random.uniform() < min(1., np.exp(-dE)):
                    tmp_state[index], tmp_state[index + direction] = tmp_state[index + direction], tmp_state[index]

        return tmp_state


    def _energy_change_of_flip(self, state, index):
        """
        Calculate the change in energy that a flip will cause.

        The energy of a site in the site is defined as
        E(site, mobile) = m * J + B
        E(site, inert) = (2 - m) * J - B

        where m is the number of mobile neighbors, J is the coupling energy and B is the inverse temperature of the system.

        :param state (np.array) state: The current state of the system.
        :param int index: The index of the site that is being examined.

        :returns float dE: The change in total energy of the state caused by flipping the site at ``index``.
        """

        if index == 0:
            dE = (-1)**self.identity_func(index, index+1, state) * self.coupling_energy + (-1)**state[index] * self.external_field
        elif index == self.num_sites - 1:
            dE = (-1)**self.identity_func(index, index-1, state) * self.coupling_energy + (-1)**state[index] * self.external_field
        else:
            dE = ((-1)**self.identity_func(index, index + 1, state) + (-1)**self.identity_func(index, index - 1, state)) * self.coupling_energy \
                + (-1)**state[index] * self.external_field

        return dE

    def _energy_change_of_swap(self, state, index, direction):
        """
        Calculate the energy chage caused by swapping two sites in the chain. Because we are dealing with a one dimensional chain
        we can heavily rely on the symmetry of the system to derive the energy change.

        The energy of a site in the site is defined as
        E(site, mobile) = m * J + B
        E(site, inert) = (2 - m) * J - B

        Only the following configurations yield a change in energy.

        [1][0][1][0] -> +4J
        [1][1][0][0] -> -4J

        All other configurations don't change the energy of the system

        :param state (np.array) state: The current state of the system.
        :param int index: The index of the site that is being examined.
        :param int direction: the direction of a swap. -1 is left and +1 is right.

        :returns float dE: The change in total energy of the state caused by flipping the site at ``index``.
        """
        # TODO: correctly handle the edges of the state vector
        # Right now if we are at risk of being at the edge we return - ln(0.5) such that the probability of swapping is 0.5
        if index >= (len(state) - 2) or index < 2:
            return 0.6314718056

        if state[index] == state[index + 2*direction] and state[index - 1] == state[index + 1] and state[index] != state[index + direction]:
            dE = 4 * self.coupling_energy
        elif state[index] == state[index - direction] and state[index + direction] == state[index + 2*direction] and state[index] != state[index + direction]:
            dE = -4 * self.coupling_energy
        else:
            dE = 0

        return dE


    def _check_neighbor_constraint(self, state, index):
        """
        Check if the neighbor constraint is met. The neighbor contstraint says that a site may only perform a move iff it has at least
        ``self.neighbor_constraint`` neighbors that are mobile.

        :param state (np.array) state: The current state of the system.
        :param int index: The index of the site that is being examined.

        :returns bool:
        """

        if index == 0:
            return state[index + 1] == 1
        elif index == len(state) - 1:
            return state[index - 1] == 1
        else:
            return state[index - 1] + state[index + 1] >= self.neighbor_constraint

    def identity_func(self, i, j, state):
        """
        The identity function. It checks if the site at index `i` is equal to the site ate index `j`

        :param state (np.array) state: The current state of the system.
        :param int i: one of the two indices of the sites that are compared
        :param int j: the other of the two indices of the sites that are compared

        :returns int: 1 if i==j and 0 if i!=j
        """
        return int(state[i] == state[j])

def draw_trajectory(trajectory):
    """
    :param (np.array) trajectory: A trajectory of shape
        (self.num_steps, self.num_sites). The rows are states.
    """
    from matplotlib.colors import ListedColormap

    plt.figure(figsize=(7,5))
    cmap = ListedColormap(['w', 'r'])
    plt.matshow(trajectory.T, cmap=cmap, fignum=1, aspect='auto')
    plt.ylabel("Site")
    plt.xlabel("Time step")
    plt.title("Trajectory")

    plt.show()


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

def east_parameter_search(probs):

    assert len(probs) > 0

    activations = []

    for prob in tqdm(probs):
        east_kcm = EastKCM(prob_transition=prob, num_burnin_steps=0, num_sites=60, num_steps=600)
        trajectory = east_kcm.gen_trajectory()
        tps = TransitionPathSampler(east_kcm, east_kcm.activity)
        activations.append(tps.mc_average(100))

    plt.figure()
    for i in range(len(probs)):
        plt.scatter(x=probs, y=activations)
    plt.xlabel("transition probability")
    plt.ylabel("average activation of 100 samples")
    plt.show()



if __name__ == "__main__":
    steps = 50
    east_parameter_search([0 + 2e-3/steps*i for i in range(steps)])
