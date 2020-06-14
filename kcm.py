# ------------------------------------------------------------

# Code to implement Kinetically Constrained Models
#
# *References*
# [1] The Fredrickson-Aldersen Model [Frederickson and Alderson 1984](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.53.1244)
# [2] The East Model [Jackle and Eisinger 1991](https://link.springer.com/article/10.1007/BF01453764)
# [3] [Nicholas B. Tutto 2018](https://tps.phys.tue.nl/lm-live/fa-ising/)

# ------------------------------------------------------------

import logging

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

        for i in range(self.num_sites - 1):
            if state[i + 1] == 1 and np.random.uniform() < self.prob_transition:
                state[i] = 1 - state[i]

        return state

class SoftenedFA(KCM):

    """

    The one-spin softened Fredrickson-Andersen Model is a kinetic 1-dimensional Ising chain.
     - It's spins are non-interacting.
     - Boundaries are periodic.

    For convenience sake, we interpret the spins as occupation numbers $$0, 1$$
    rather than the conventional up/down $$\pm 1$$.

    This class produces a single trajectory.

    # Params
    s is the biasing_field
    gamma is how much faster a mobile site deactivates (in a mobile region) than an immobile site activates (in the same region)
    epsilon is the softening parameter: it determines to what extent sites that are in an immobile region can still activate/deactivate

    # Rates
    The rate for flipping a spin 0 -> 1 is ``= C_i``
    The rate for flipping a spin 1 -> 0 is ``= gamma * C_i``
    The rate of swapping (0, 1) -> (1,0) is ``= rate_swap``

    where ``C_i = \sum_{j \in nn(i)} [n_j+ (\epsilon/2)]``

    This entire class is built off of the description of Elmatad et al. 2010, first column of the second page.

    """

    def __init__(self, s=None, eps=0., gamma=1., num_sites=50, num_steps=100, num_burnin_steps=0):
        """
        :param float s: The biasing field that determines the rate of swaps. Defaults to the critical biasing field value.
        :param float eps: The amount of softening; this allows immobile regions a small probability of becoming mobile
        :param float gamma: The relative rate of inactivation vs activation in the mobile region.
        :Param int num_sites: The number of sites to include in the chain.
        :param int num_steps: The number of steps to include in a trajectory.
        :param int num_burnin_steps: The number of steps to equilibrate a
            randomly initialized configuration.

        """
        super(SoftenedFA, self).__init__(0., num_sites=num_sites, num_steps=num_steps, num_burnin_steps=num_burnin_steps)

        # MODEL PARAMETERS
        self.eps = eps # This captures "activation energy" (U) and "temperature" (T) ~ exp(-U/T)
        self.gamma = gamma # This captures "coupling energy" (J) and "temperature" (J) ~ exp(-J/T)

        if s is None:
            s = self.get_critical_s()

        self.s = s

        # TRANSITION RATES

        # As in Elmatad et al., we have chosen to set lambda, the overall rate of the fast processes to be equal to 1.

        self.rate_swap = self.get_rate_swap_from_s(s)
        self.get_neighbor_constraint = lambda neighbors: (np.sum(neighbors) + np.size(neighbors) * self.eps / 2) # C_i in Elmatad et al.
        self.get_rate_activation = lambda neighbors:  self.get_neighbor_constraint(neighbors)
        self.get_rate_inactivation = lambda neighbors:  (self.gamma * self.get_neighbor_constraint(neighbors))

        # TRANSITION PROBABILITIES

        # Elmatad et al. describes these processes in terms of rates.
        # To get probability that any timestep a transition occurs from rates we use the formula

        # DEBUGGING

        # Just to make sure that the determined rates have the right order of magnitude

        logging.info("Neighboring spins; Rate activation; Rate inactivation")
        for neighbors in [np.array([1, 1]), np.array([1,0 ]), np.array([0, 1]), np.array([0, 0])]:
            logging.info("Neighbors {}".format(neighbors))
            logging.info("Rate Activation {}".format(self.get_rate_activation(neighbors)))
            logging.info("Rate Deactivation {}".format(self.get_rate_inactivation(neighbors)))

        logging.info("Biasing field: {}".format(self.s))
        logging.info("Rate swap: {}".format(self.rate_swap))

    def get_rate_swap_from_s(self, s):
        # Formula 4 in Elmatad et al.
        return (1. - self.gamma + np.sqrt(np.square(1. - self.gamma) + 4. * np.exp(-s) * self.gamma)) / 2.

    def get_critical_s(self):
        # Shorthand for convenience:
        D = self.get_rate_swap_from_s

        # Formula 5 in Elmatad et al.:
        lhs = lambda s: (1. - self.gamma) / (1. + self.eps)
        rhs = lambda s: np.sqrt((1. - self.gamma - D(s) * np.square(1 - np.exp(-s))) +4. * np.exp(-2. * s) * self.gamma) - (1. - np.exp(-s) * D(s))

        # We use a root-solver to find values of s such that this equality holds
        # Since the right-hand-side minus the left-hand-side should equal 0,
        # where we compute D as a function of s.
        sol = optimize.root_scalar(lambda s: rhs(s) - lhs(s), x0=0, x1=.1)

        # TODO: In Elmatad et al., the derived values are in the order of 0.0001-0.01
        # However, this method seems to produce values around .3-.5
        return sol.root

    @staticmethod
    def _get_neighbors(index, state):
        neighbors = []

        # NOTE: We are usingperiodic boundary conditions
        if index == 0:
            neighbors = [state[index + 1], state[len(state) - 1]]
        elif index == len(state) - 1:
            neighbors = [state[index - 1], state[0]]
        else:
            neighbors = [state[index - 1], state[index + 1]]

        return np.array(neighbors)

    @staticmethod
    def _flip(state, i):
        """
        Maps 0 -> 1 or 1 -> 0 at index ``i`` in ``state``
        """
        state[i] = 1 - state[i]
        return state

    def _swap(self, state, i):
        """
        Flips spins ``i`` and ``i+1``.

        Since we are using periodic boundary conditions,
        if ``i`` is the last index in state, ``i+1`` is the first index.
        """
        state[i], state[(i + 1) % self.num_sites] = state[(i + 1) % self.num_sites], state[i]
        return state

    def _step(self, state):

        """
        The probability of switching to a particular state C' is

        P(C -> C') = W(C -> C') / r(C),
        where W(C -> C') is the rate of switching to that state C', and
        where r(C) = sum_{C'} W(C -> C') is the sum of all transition rates

        The set of possible C' given C are all sites within one flip or swap of C.

        See (pg 2. top-left): https://www.researchgate.net/publication/7668956_Chaotic_Properties_of_Systems_with_Markov_Dynamics
        """
        i = np.random.choice(np.arange(self.num_sites, dtype=np.int32))
        neighbors = self._get_neighbors(i, state)

        swap_weight = 0
        flip_weight = 0

        if state[i] == 0:
            # For an immobile state we are interested in the possibilitie of activation.

            flip_weight = self.get_rate_activation(neighbors)

            # There is also the possibility of swapping [0, 1] -> [1, 0]
            if  state[(i + 1) % self.num_sites] == 1:
                swap_weight = self.rate_swap

        else:
            flip_weight = self.get_rate_inactivation(neighbors)

            # There is also the possibility of swapping [1, 0] -> [0, 1]
            if state[(i + 1) % self.num_sites] == 0:
                swap_weight = self.rate_swap

        total_weight = swap_weight + flip_weight

        # Step 2: Decide whether or not to update the state
        # NOTE: This is the step I am most uncertain about!!
        if (np.random.uniform() > np.exp(-total_weight)):
            return state

        # Step 3: If we update, decide whether to flip or swap
        flip_prob = flip_weight / total_weight
        swap_prob = swap_weight / total_weight

        choice = np.random.uniform()

        if choice < flip_prob:
            state[i] = 1 - state[i]
        else:
            state[i], state[(i + 1) % self.num_sites] = state[(i + 1) % self.num_sites], state[i]

        return state

    def activity(self, trajectory):
        activity = 0.
        for i in range(self.num_steps - 1):
            if not np.isclose(trajectory[i, :], trajectory[i + 1, :]).all():
                activity += 1.

        # NOTE: Elmatad et al. mention dividing by N t_obs, but that doesn't really make sense
        # when there is only one local change in successive configurations
        return activity / (self.num_steps) # This is the intensive activity

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

    cmap = ListedColormap(['r', 'w'])
    plt.matshow(trajectory.T, cmap=cmap,)
    plt.ylabel("Site")
    plt.xlabel("Time step")
    plt.title("Trajectory")

    plt.show()


if __name__ == "__main__":
    fa_kcm = OneSpinFAKCM(0.5, 0.5, 25, 200, 0.1, 0.1, 1, 50000)
    trajectory = fa_kcm.gen_trajectory()
    draw_trajectory(trajectory)
