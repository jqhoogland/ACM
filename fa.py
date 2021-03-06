import logging

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from tqdm.notebook import tqdm

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

    def _energy(self, state):
        return -np.log(self.gamma) * np.sum(state)

    def _accept_state(self, state, new_state):
        return np.random.uniform() < min(1, np.exp(self._energy(state) - self._energy(new_state)))

    def _equilibrate(self, state, num_equilibration_steps, draw=True):
        evolution = np.zeros((num_equilibration_steps, state.size))

        def step(state):
            i = np.random.choice(np.arange(state.size))
            state[i] = 1 - state[i]
            return state

        for i in range(num_equilibration_steps):
            tmp_state = np.copy(state)
            tmp_state = step(tmp_state)

            # Does this need to incorporate gamma in some way?
            if self._accept_state(state, tmp_state):
                state = tmp_state

            evolution[i, :] = state[:]

        if draw:
            draw_discrete_trajectory(evolution, "Thermodynamic equilibration of the initial spatial configuration", "Time step", "Site")
            draw_average_plot(np.mean(evolution, axis=1), "Ratio of active to inactive sites over time", "Time step", "Ratio active:inactive sites")

        return state

    def gen_trajectory(self, init_state=None, num_equilibration_steps=1000, draw=True):
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

        if init_state is None:
            state = np.random.choice([0, 1], size=(self.num_sites, ))
            state = self._equilibrate(state, num_equilibration_steps, draw)

        assert state.shape == (self.num_sites,)

        burnin_trajectory = np.zeros((self.num_burnin_steps, self.num_sites))
        burnin_occupation_times = np.zeros(self.num_burnin_steps)
        for i in range(self.num_burnin_steps - 1): # -1 because we update state once more in the forloop below.
            state, time = self._step(state)
            burnin_trajectory[i, :] = state
            burnin_occupation_times[i] = time

        trajectory = np.zeros((self.num_steps, self.num_sites))
        occupation_times = np.zeros(self.num_steps)
        for i in range(self.num_steps):
            state, time = self._step(state)
            trajectory[i, :] = state
            occupation_times[i] = time

        if draw:
            draw_trajectory(burnin_trajectory, burnin_occupation_times, "Burn-in period")
            draw_trajectory(trajectory, occupation_times, "Trajectory")

        return trajectory, occupation_times

    def fixed_time_activity(self, trajectory, occupation_times, time_period):
        i = 0
        time = occupation_times[0]

        while time < time_period and i < len(occupation_times) - 1:
            i += 1
            time += occupation_times[i]

        activity = time_period * 1. / i

        if i == len(occupation_times) - 1:
            logging.warning("Fixed time {} exceeded limit {}.".format(time_period, np.sum(occupation_times)))
            activity = self.activity(trajectory, occupation_times)

        return activity

    def activity(self, trajectory, occupation_times):
        return 1. * len(trajectory) / (self.num_sites * np.sum(occupation_times))

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
        self.get_neighbor_constraint = lambda neighbors: (np.sum(neighbors) + self.eps) # C_i in Elmatad et al.
        self.get_rate_activation = lambda neighbors: self.gamma * self.get_neighbor_constraint(neighbors)
        self.get_rate_inactivation = lambda neighbors:  self.get_neighbor_constraint(neighbors)
        self.get_rate_swap = lambda site, neighbors: float(site != neighbors[-1]) * self.rate_swap

        # TRANSITION PROBABILITIES

        # Elmatad et al. describes these processes in terms of rates.
        # To get probability that any timestep a transition occurs from rates we use the formula

        # DEBUGGING

        # Just to make sure that the determined rates have the right order of magnitude

        logging.info("Neighboring spins; Rate activation; Rate inactivation")
        for neighbors in [np.array([1, 1]), np.array([1,0 ]), np.array([0, 1]), np.array([0, 0])]:
            logging.info("Rate Flip (activation) [{0}, 0, {1}]-> [{0}, 1, {1}]: {2}".format(neighbors[0], neighbors[ -1], self.get_rate_activation(neighbors)))
            logging.info("Rate Flip (deaactivation)[{0}, 1, {1}]-> [{0}, 0, {1}]: {2} ".format(neighbors[0], neighbors[ -1], self.get_rate_inactivation(neighbors)))
            logging.info("Rate swap [{0}, 0, {1}] -> [{0}, {1}, 0]: {2}".format(neighbors[0], neighbors[-1], self.get_rate_swap(0, neighbors)))
            logging.info("Rate swap [{0}, 1, {1}] -> [{0}, {1}, 1]: {2}".format(neighbors[0], neighbors[-1], self.get_rate_swap(1, neighbors)))

        logging.info("Biasing field: {}".format(self.s))
        logging.info("Rate swap: {}".format(self.rate_swap))

    def get_rate_swap_from_s(self, s):
        # Formula 4 in Elmatad et al.
        return (1. - self.gamma + np.sqrt(np.square(1. - self.gamma) + 4. * np.exp(-s) * self.gamma)) / 2.

    def get_critical_s(self):
        # Shorthand for convenience:
        D = self.get_rate_swap_from_s

        # Formula 5 in Elmatad et al.:
        lhs = lambda s: (1. +self.gamma) / (1. + self.eps)
        rhs = lambda s: np.sqrt(np.square(1. - self.gamma - D(s) * (1 - np.exp(-s))) +4. * np.exp(-2. * s) * self.gamma) - (1. - np.exp(-s) * D(s))

        # We use a root-solver to find values of s such that this equality holds
        # Since the right-hand-side minus the left-hand-side should equal 0,
        # where we compute D as a function of s.
        sol = optimize.root_scalar(lambda s: rhs(s) - lhs(s), x0=0, x1=.1)

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

            else: # The state is active
                flip_rates[i] = self.get_rate_inactivation(neighbors)

            # There is also the possibility of swapping [1, 0] -> [0, 1]
            swap_rates[i] = self.get_rate_swap(state[i], neighbors)

        return swap_rates, flip_rates

    def get_transition_times(self, rates):
        """
        Creates an array of exponentially distributed times for each rate in rates.

        i.e. inverts f(x; r) to get x where

        f(x; r) = r exp(-r x)

        is the pdf, and r is the rate.

        Then the cdf is

        c(x; r) = 1 - exp(-rx)

        with inverse

        x = - np.log(1 - c(x; r)) / r
        """

        times = -np.log(1 - np.random.uniform(size=rates.size)) / rates
        times[times <= 0] = np.inf

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

        return trajectory, occupation_times

    def _shift(self, trajectory, occupation_times):
        i = np.random.choice(np.arange(self.kcm.num_sites, dtype=np.int32))

        trajectory[:self.kcm.num_steps - i, :] = trajectory[i:, :]
        occupation_times[:self.kcm.num_steps - i] = occupation_times[i:]

        for j in range(self.kcm.num_steps - i, self.kcm.num_steps -1):
            trajectory[j + 1, :], occupation_times[j + 1]= self.kcm._step(trajectory[j, :])

        return trajectory, occupation_times

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

    def mc_samples(self, num_samples, num_burnin_steps=0, draw=False, draw_every=10):
        # TODO: also measure the standard deviations (or do a fancier binning analysis)!

        # Prepare the initial trajectory
        trajectory, occupation_times= self.kcm.gen_trajectory(draw=draw)

        # Allocate space for the trial trajectory
        trial_trajectory = np.copy(trajectory)
        trial_occupation_times = np.copy(occupation_times)

        # We will associate an "energy" to each trajectory, and use to
        # sample the trajectory space according to a Metropolis-Hastings condition
        energy = self.get_trajectory_weight(trajectory, occupation_times)

        # First, we proceed with a number of burn-in sequences.
        # In principle, this is unnecessary, since we have already accomplished
        # this in generating the trajectory.
        burnin_measurements = np.zeros(num_burnin_steps)
        for i in tqdm(range(num_burnin_steps), desc="Equilibrating TPS"):
            burnin_measurements[i] = self.observable(trajectory, occupation_times)

            # Copy the trajectory into trial first so we don't change trial in place
            trial_trajectory[:, :] = trajectory[:, :]
            trial_occupation_times[:] = occupation_times[:]
            trial_trajectory, trial_occupation_times = self._step(trial_trajectory, trial_occupation_times)
            trial_energy = self.get_trajectory_weight(trial_trajectory, trial_occupation_times)

            if self.accept_trajectory(energy, trial_energy):
                trajectory[:, :] = trial_trajectory[:, :]
                occupation_times[:]= trial_occupation_times[:]
                energy = trial_energy

            if draw and (i + 1) % draw_every == 0:
                draw_trajectory(trajectory, occupation_times, title="Burn-in iteration {}".format(i + 1))

        proportion_active = np.zeros(num_samples)
        measurements = np.zeros(num_samples)
        for i in tqdm(range(num_samples), desc="Generating TPS samples"):
            measurements[i] = self.observable(trajectory, occupation_times)
            proportion_active[i] = np.mean(discretize_trajectory(trajectory, occupation_times))

            # Copy the trajectory into trial first so we don't change trial in place
            trial_trajectory[:, :] = trajectory[:, :]
            trial_occupation_times[:] = occupation_times[:]
            trial_trajectory, trial_occupation_times= self._step(trial_trajectory, trial_occupation_times)
            trial_energy = self.get_trajectory_weight(trial_trajectory, trial_occupation_times)

            if self.accept_trajectory(energy, trial_energy):
                trajectory[:, :] = trial_trajectory[:, :]
                occupation_times[:]= trial_occupation_times[:]
                energy = trial_energy

            if draw and (i + 1) % draw_every == 0:
                draw_trajectory(trajectory, occupation_times, title="TPS sample {}".format(i + 1))

        if draw:
            draw_average_plot(burnin_measurements, "Activity over TPS Burn-in Period", "TPS step", "Activity")
            draw_average_plot(measurements, "Activity over TPS Samples", "TPS step", "Activity")
            draw_average_plot(proportion_active, "Ratio of active to inactive sites over time", "Time step", "Ratio active:inactive sites")

        return measurements

    def mc_average(self, num_samples, num_burnin_steps=0, draw=False, draw_every=10):
        return np.mean(self.mc_samples(num_samples, num_burnin_steps, draw, draw_every))

class SoftenedFATPS(TransitionPathSampler):
    def __init__(self, *args):
        """
        docstring
        """
        super(SoftenedFATPS, self).__init__(*args)
        self.alpha = np.arctan(2. * np.exp(-self.kcm.s) * (1. + self.kcm.eps) / (2. +self.kcm.eps - self.kcm.eps * self.kcm.gamma))
        self.g = np.log(np.tan(self.alpha / 2.))

        print("alpha {}, g {}".format(self.alpha, self.g))


    def get_trajectory_weight(self, trajectory, occupation_times):
        #print(self.g, np.sum(trajectory[0, :] + trajectory[-1, :]))
        return self.kcm.s * self.kcm.activity(trajectory, occupation_times) * self.kcm.num_sites * np.sum(occupation_times) - self.g * (np.sum(trajectory[0, :] + trajectory[-1, :]))

    @staticmethod
    def accept_trajectory(energy_prev, energy_curr):
        """ The simple Metropolis condition
        """
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


def draw_discrete_trajectory(trajectory, title, xlabel, ylabel, time_step=1.):
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(['w', 'r'])
    fig, ax= plt.subplots()
    ax.matshow(trajectory.T, cmap=cmap, aspect='auto')
    ax.set_title(title, pad=20)
    ax.set_xticklabels((np.arange(-1, len(trajectory)) * time_step))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.show()

def draw_trajectory(trajectory, occupation_times, title="Trajectory", time_step=0.01):
    """
    :param (np.array) trajectory: A trajectory of shape
        (self.num_steps, self.num_sites). The rows are states.
    """
    discrete_trajectory = discretize_trajectory(trajectory, occupation_times, time_step)
    draw_discrete_trajectory(discrete_trajectory, title, "Time step", "Site", time_step)


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
