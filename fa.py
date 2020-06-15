import logging

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize

class ContinuousTimeKCM(object):
    """

    For convenience sake, we interpret the spins as occupation numbers $$0, 1$$
    rather than the conventional up/down $$\pm 1$$.

    This class produces a single trajectory.

    """

    def __init__(self, num_sites, num_steps, num_burnin_steps=0):
        """
        :param float trans_prob: The probability (with value in range [0, 1])
            of a spin flipping if its right neighbor is up (i.e. =1).
        :param int num_sites: The number of sites to include in the chain.
        :param int num_steps: The number of steps to include in a trajectory.
        :param int num_burnin_steps: The number of steps to equilibrate a
            randomly initialized configuration.

        """
        assert num_sites > 1

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
        occupation_times = np.zeros(self.num_steps)

        if init_state is None:
            state = np.random.choice([0, 1], size=(self.num_sites, ))

        assert state.shape == (self.num_sites, )

        for _i in range(self.num_burnin_steps - 1): # -1 because we update state once more in the forloop below.
            state, _t = self._step(state)

        for i in range(self.num_steps):
            state, time = self._step(state)
            trajectory[i, :] = state
            occupation_times[i] = time

        return trajectory, occupation_times

    def activity(self, trajectory, occupation_times):
        return len(trajectory) / (self.num_sites * np.sum(occupation_times))

class SoftenedFA(ContinuousTimeKCM):

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
        super(SoftenedFA, self).__init__(num_sites=num_sites, num_steps=num_steps, num_burnin_steps=num_burnin_steps)

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
        self.get_rate_activation = lambda neighbors: self.gamma * self.get_neighbor_constraint(neighbors)
        self.get_rate_inactivation = lambda neighbors:  self.get_neighbor_constraint(neighbors)

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

    def get_transition_rates(self, state):
        swap_rates = np.zeros(self.num_sites)
        flip_rates = np.zeros(self.num_sites)

        for i in range(self.num_sites):
            neighbors = self._get_neighbors(i, state)

            if state[i] == 0: # The state is inactive
                flip_rates[i] = self.get_rate_activation(neighbors)

                # There is also the possibility of swapping [0, 1] -> [1, 0]
                if state[(i + 1) % self.num_sites] == 1:
                    swap_rates[i] = self.rate_swap

            else: # The state is active
                flip_rates[i] = self.get_rate_inactivation(neighbors)

                # There is also the possibility of swapping [1, 0] -> [0, 1]
                # In Elmatad et al., they only discuss [0, 1] -> [0, 1], but
                # we implicitly need the inverse in order to satisfy detailed balance
                if state[(i + 1) % self.num_sites] == 0:
                    swap_rates[i] = self.rate_swap

        return swap_rates, flip_rates

    def get_transition_times(self, rates):
        """
        Creates an array of exponentially distributed times for each rate in rates.

        i.e. inverts f(x; r) to get x where

        f(x; r) = r exp(-r x)

        is the pdf, and r is the rate.

        i.e. x = - (1 / r) * log ( f(x; r) / r )
        """
        times = np.abs((1. / rates) * np.log(np.random.uniform(size=rates.size) / rates))

        times[times == np.inf] = 1e8

        return times

    def _step(self, state):
        """
        The probability of switching to a particular state C' is

        P(C -> C') = W(C -> C') / r(C),
        where W(C -> C') is the rate of switching to that state C', and
        where r(C) = sum_{C'} W(C -> C') is the sum of all transition rates

        The set of possible C' given C are all sites within one flip or swap of C.

        Average time in a state is 1/r(C)

        See (pg 2. top-left): https://www.researchgate.net/publication/7668956_Chaotic_Properties_of_Systems_with_Markov_Dynamics
        """
        # STEP 1. DETERMINE RATES W(C -> C')

        swap_rates, flip_rates = self.get_transition_rates(state)

        # STEP 2. DETERMINE CLOCK TIMERS

        swap_times = self.get_transition_times(swap_rates)
        flip_times = self.get_transition_times(flip_rates)


        min_swap_time_idx = np.argmin(swap_times)
        min_flip_time_idx = np.argmin(flip_times)

        min_swap_time = swap_times[min_swap_time_idx]
        min_flip_time = flip_times[min_flip_time_idx]

        time = 0

        if min_swap_time < min_flip_time:
            state = self._swap(state, min_swap_time_idx)
            time = min_swap_time
        else:
            state = self._flip(state, min_flip_time_idx)
            time = min_flip_time

        return state, time

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

    def _half_shoot(self, trajectory, occupation_times):
        i = np.random.choice(np.arange(self.kcm.num_sites, dtype=np.int32))

        for j in range(i, self.kcm.num_steps -1):
            trajectory[j + 1, :], occupation_times[j + 1] = self.kcm._step(trajectory[j, :])

        return trajectory

    def _shift(self, trajectory, occupation_times):
        i = np.random.choice(np.arange(self.kcm.num_sites, dtype=np.int32))

        trajectory[:self.kcm.num_steps - i, :] = trajectory[i:, :]
        occupation_times[:self.kcm.num_steps - i] = occupation_times[i:]

        for j in range(self.kcm.num_steps - i, self.kcm.num_steps -1):
            trajectory[j + 1, :], occupation_times[j + 1]= self.kcm._step(trajectory[j, :])

        return trajectory

    def _step(self, trajectory, occupation_times):
        # With equal probability we either half-shoot or shift.
        if np.random.uniform() < 0.5:
            return self._half_shoot(trajectory, occupation_times)
        else:
            return self._shift(trajectory, occupation_times)

    def get_trajectory_weight(self, trial_trajectory, occupation_times):
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

        trajectory, occupation_times= self.kcm.gen_trajectory()

        energy_prev = self.get_trajectory_weight(trajectory, occupation_times)
        energy_curr = energy_prev

        for i in range(num_samples):
            measurements[i] = self.observable(trajectory, occupation_times)

            if verbose:
                draw_trajectory(trajectory, occupation_times)

            trial_trajectory = self._step(trajectory, occupation_times)

            energy_prev = energy_curr
            energy_curr = self.get_trajectory_weight(trial_trajectory, occupation_times)

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

def discretize_trajectory(trajectory, occupation_times, time_step=0.01):
    trajectory_duration = np.sum(occupation_times)
    num_transitions, num_sites = trajectory.shape
    num_discrete_steps = int(trajectory_duration / time_step)
    discrete_trajectory = np.zeros([num_discrete_steps, num_sites])

    j = 0 # index which goes over the number of discretized steps
    for i in range(num_transitions): # index which goes over the number of original steps
        time_in_state = occupation_times[i]
        num_discrete_steps_in_state = int(time_in_state / time_step)
        if num_discrete_steps_in_state > 0:
            discrete_trajectory[j: j + num_discrete_steps_in_state, :] = trajectory[i, :]

        j += num_discrete_steps_in_state

    return discrete_trajectory


def draw_trajectory(trajectory, occupation_times, time_step=0.01):
    """
    :param (np.array) trajectory: A trajectory of shape
        (self.num_steps, self.num_sites). The rows are states.
    """
    from matplotlib.colors import ListedColormap

    discrete_trajectory = discretize_trajectory(trajectory, occupation_times, time_step)

    cmap = ListedColormap(['r', 'w'])
    plt.matshow(discrete_trajectory.T, cmap=cmap,)
    plt.ylabel("Site")
    plt.xlabel("Time step")
    plt.title("Trajectory")

    plt.show()
