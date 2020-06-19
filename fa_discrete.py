import logging

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from tqdm.notebook import tqdm

class DiscreteTimeKCM(object):
    """

    For convenience sake, we interpret the spins as occupation numbers $$0, 1$$
    rather than the conventional up/down $$\pm 1$$.

    This class produces a single trajectory.

    """

    def __init__(self, num_sites, num_steps, num_burnin_steps=0, time_step=0.001):
        """
        :param int num_sites: The number of sites to include in the chain.
        :param int num_steps: The number of steps to include in a trajectory.
        :param int num_burnin_steps: The number of steps to equilibrate a
            randomly initialized configuration.
        :param float time_step: The duration of the discretized time step

        """
        assert num_sites > 1

        self.num_sites = num_sites
        self.num_steps = num_steps
        self.num_burnin_steps = num_burnin_steps
        self.time_step = time_step

    def _step(self, state):
        raise NotImplementedError("")

    def gen_init_state(self):
        """
        Generates a state with a proportion of active spins equal, on average,
        to ``self.equilibrium_activation``

        """

        return np.random.choice(
            [0, 1],
            size=(self.num_sites, ),
            p=[(1. - self.equilibrium_activation), self.equilibrium_activation]
        )

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
            state = self.gen_init_state()

        assert state.shape == (self.num_sites,)

        burnin_trajectory = np.zeros((self.num_burnin_steps, self.num_sites))
        for i in range(self.num_burnin_steps - 1): # -1 because we update state once more in the forloop below.
            state = self._step(state)
            burnin_trajectory[i, :] = state

        trajectory = np.zeros((self.num_steps, self.num_sites))
        for i in range(self.num_steps):
            state = self._step(state)
            trajectory[i, :] = state

        if draw:
            draw_trajectory(burnin_trajectory, "Burn-in period")
            draw_trajectory(trajectory, "Trajectory")

        return trajectory

    def activity_intensive(self, trajectory):
        return self.activity(trajectory) / (self.num_sites * len(trajectory) * self.time_step)

    def activity(self, trajectory):
        activity = 0

        for i in range(1, len(trajectory)):
            if not (trajectory[i]== trajectory[i - 1]).all():
                activity += 1

        return activity


class SoftenedFA(DiscreteTimeKCM):

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

    def __init__(self, s=None, rate_swap=None, eps=0., gamma=1., num_sites=50, num_steps=100, num_burnin_steps=0, time_step=0.005):
        """
        :param float s: The biasing field that determines the rate of swaps. Defaults to the critical biasing field value.
        :param float eps: The amount of softening; this allows immobile regions a small probability of becoming mobile
        :param float gamma: The relative rate of inactivation vs activation in the mobile region.
        :Param int num_sites: The number of sites to include in the chain.
        :param int num_steps: The number of steps to include in a trajectory.
        :param int num_burnin_steps: The number of steps to equilibrate a
            randomly initialized configuration.

        """
        super(SoftenedFA, self).__init__(num_sites=num_sites, num_steps=num_steps, num_burnin_steps=num_burnin_steps, time_step=time_step)

        self.eps = eps # This captures "activation energy" (U) and "temperature" (T) ~ exp(-U/T)
        self.gamma = gamma # This captures "coupling energy" (J) and "temperature" (J) ~ exp(-J/T)


        # $\gamma = c / (1 - c )$ where $c$ is the equilibrium concentration of active spins.
        # This is needed to initialize the first configuration in the trajectory
        self.equilibrium_activation = self.gamma / (1 + self.gamma)

        # s and rate_swap are related to one another by formula 4 in Elmatad et al.
        # We can provide either to derive the other.
        # If both are provided, we use rate_swap, and derive s from the rate_swap.

        if s is None:
            s = self.get_critical_s()

        if rate_swap is None:
            rate_swap = self.get_rate_swap_from_s(s)
        else:
            s = self.get_s_from_rate_swap(rate_swap)

        self.s = s
        self.rate_swap = rate_swap

        # Functions to derive Transition Rates from a spin and its neighbors
        self.get_neighbor_constraint = lambda neighbors: (np.sum(neighbors) + self.eps) # C_i in Elmatad et al.
        self.get_rate_activation = lambda neighbors: self.gamma * self.get_neighbor_constraint(neighbors)
        self.get_rate_inactivation = lambda neighbors:  self.get_neighbor_constraint(neighbors)
        self.get_rate_swap = lambda site, neighbors: float(site != neighbors[-1]) * self.rate_swap

        # The g term is needed to counter the effects of closed time boundaries
        self.z = np.exp(-self.s)
        self.alpha = np.arctan(2. * self.z * np.sqrt(self.gamma) / ( - self.rate_swap * (1 - self.z))) / 2.

        self.g = np.log(np.abs(np.tan(self.alpha / 2.)))

        print("alpha {}, g {}".format(self.alpha, self.g))

        # DEBUGGING
        logging.info("Biasing field: {}".format(self.s))
        logging.info("Rate swap: {}".format(self.rate_swap))

        for neighbors in [np.array([1, 1]), np.array([1,0 ]), np.array([0, 1]), np.array([0, 0])]:
            logging.info("Rate Flip (activation) [{0}, 0, {1}]-> [{0}, 1, {1}]: {2}".format(neighbors[0], neighbors[ -1], self.get_rate_activation(neighbors)))
            logging.info("Rate Flip (deaactivation)[{0}, 1, {1}]-> [{0}, 0, {1}]: {2} ".format(neighbors[0], neighbors[ -1], self.get_rate_inactivation(neighbors)))
            logging.info("Rate swap [{0}, 0, {1}] -> [{0}, {1}, 0]: {2}".format(neighbors[0], neighbors[-1], self.get_rate_swap(0, neighbors)))
            logging.info("Rate swap [{0}, 1, {1}] -> [{0}, {1}, 1]: {2}".format(neighbors[0], neighbors[-1], self.get_rate_swap(1, neighbors)))


    def get_s_from_rate_swap(self, rate_swap):
        # Formula 4. in Elmatad et al. inverted to get s from D
        arg = (np.square(2 * rate_swap + self.gamma -1) - np.square(1 - self.gamma)) / (4. * self.gamma)
        if arg == 0:
            return np.inf
        return -np.log(arg)

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
        """
        We use periodic boundaries, so the first and last spins are also neighbors.
        """
        return np.array([state[(index - 1) % len(state)], state[(index + 1) % len(state)]])

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
        """
        For a state of ``self.num_sites`` spins, there are ``2 * self.num_sites``
        possible transitions.
        - ``self.num_sites`` possible sites to flip
        - ``self.num_sites`` possible pairs to swap

        If a transition is prohibited, then we assign a transition rate of 0.

        :returns (swap_rates, flip_rates): the respect transition rates for swaps
        and flips at each of these sites.

        """
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
        Creates an array of exponentially distributed times, $t$ for each rate,
        $r$ in ``rates``.

        $PDF(t; r) = r exp(-r t)$
        $CDF(t; r) = 1 - exp(-rt)$

        with inverse
        $t = - np.log(1 - CDF(t; r)) / r$

        If we uniformly generate a $CDF(t; r) \in [0, 1]$, we can use this inverted
        formula to get a suitable time $t$

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

        # STEP 3. DETERMINE THE MINIMUM TRANSITION TIME (of both swaps and flips)
        min_swap_time_idx = np.argmin(swap_times)
        min_flip_time_idx = np.argmin(flip_times)

        min_swap_time = swap_times[min_swap_time_idx]
        min_flip_time = flip_times[min_flip_time_idx]

        # STEP 4. UPDATE (OR NOT) THE STATE
        # If the minimum transition time is shorter than our discrete timestep,
        # we update the state, otherwise we leave our state as is.
        if min_swap_time < min_flip_time and min_swap_time < self.time_step:
            state = self._swap(state, min_swap_time_idx)
        elif min_flip_time < self.time_step:
            state = self._flip(state, min_flip_time_idx)

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
        """
        Choose a random index of the trajectory, and remainder of the trajectory
        starting at that index.
        """
        i = np.random.choice(np.arange(self.kcm.num_sites, dtype=np.int32))

        for j in range(i, self.kcm.num_steps -1):
            trajectory[j + 1, :] = self.kcm._step(trajectory[j, :])

        return trajectory

    def _shift(self, trajectory):
        """
        Choose a random index of the trajectory as the new starting point,
        and extend the end of the trajectory until you reach a trajectory of the
        same duration.
        """
        i = np.random.choice(np.arange(self.kcm.num_sites, dtype=np.int32))

        trajectory[:self.kcm.num_steps - i, :] = trajectory[i:, :]

        for j in range(self.kcm.num_steps - i, self.kcm.num_steps -1):
            trajectory[j + 1, :] = self.kcm._step(trajectory[j, :])

        return trajectory

    def _step(self, trajectory):
        """
        We use half-shooting or shifting with equal likelihood.
        """
        if np.random.uniform() < 0.5:
            return self._half_shoot(trajectory)
        else:
            return self._shift(trajectory)

    def get_trajectory_weight(self, trial_trajectory):
        raise NotImplementedError

    @staticmethod
    def accept_trajectory(energy_prev, energy_curr):
        """
        The simple Metropolis-Hastings condition.
        """
        return np.random.uniform() < min(1, np.exp(energy_prev - energy_curr))

    def mc_samples(self, num_samples, num_burnin_steps=0, draw=False, draw_every=10):
        # TODO: also measure the standard deviations (or do a fancier binning analysis)!

        # Prepare the initial trajectory
        trajectory= self.kcm.gen_trajectory(draw=draw)

        # Allocate space for the trial trajectory
        trial_trajectory = np.copy(trajectory)

        # We will associate an "energy" to each trajectory, and use to
        # sample the trajectory space according to a Metropolis-Hastings condition
        # This ``get_trajectory_weight`` is implemented in child classes.
        energy = self.get_trajectory_weight(trajectory)

        # STEP 1. EQUILIBRATE THE TRAJECTORY
        # We also make measurements of the observable through this period so we
        # can verify that equilibration is taking place.
        burnin_measurements = np.zeros(num_burnin_steps)
        for i in tqdm(range(num_burnin_steps), desc="Equilibrating TPS"):
            burnin_measurements[i] = self.observable(trajectory)

            # We copy the trajectory into the trial since numpy updates matrices
            # in-place.
            trial_trajectory[:, :] = trajectory[:, :]
            trial_trajectory = self._step(trial_trajectory)
            trial_energy = self.get_trajectory_weight(trial_trajectory)

            if self.accept_trajectory(energy, trial_energy):
                trajectory[:, :] = trial_trajectory[:, :]
                energy = trial_energy

            if draw and (i + 1) % draw_every == 0:
                draw_trajectory(trajectory, title="Burn-in iteration {}".format(i + 1))

        proportion_active = np.zeros(num_samples)
        measurements = np.zeros(num_samples)
        for i in tqdm(range(num_samples), desc="Generating TPS samples"):
            measurements[i] = self.observable(trajectory)
            proportion_active[i] = np.mean(trajectory)

            # We copy the trajectory into the trial since numpy updates matrices
            # in-place.
            trial_trajectory[:, :] = trajectory[:, :]
            trial_trajectory= self._step(trial_trajectory)
            trial_energy = self.get_trajectory_weight(trial_trajectory)

            if self.accept_trajectory(energy, trial_energy):
                trajectory[:, :] = trial_trajectory[:, :]
                energy = trial_energy

            if draw and (i + 1) % draw_every == 0:
                draw_trajectory(trajectory, title="TPS sample {}".format(i + 1))

        if draw:
            draw_average_plot(burnin_measurements, "Activity over TPS Burn-in Period", "TPS step", "Activity")
            draw_average_plot(measurements, "Activity over TPS Samples", "TPS step", "Activity")
            draw_average_plot(proportion_active, "Ratio of active to inactive sites over time", "Time step", "Ratio active:inactive sites")

        return measurements

    def mc_average(self, num_samples, num_burnin_steps=0, draw=False, draw_every=10):
        return np.mean(self.mc_samples(num_samples, num_burnin_steps, draw, draw_every))

    def error_analysis(self, measurements):
        """
        Perform a binning analysis of the error in measurements.
        This is needed since the measurements are auto-correlated.

        TODO: implement this.
        """
        return np.std(measurements)

    def mc_analysis(self, num_samples, num_burnin_steps=0, draw=False, draw_every=10):
        measurements = (self.mc_samples(num_samples, num_burnin_steps, draw, draw_every))
        return np.mean(measurements), self.error_analysis(measurements)

class SoftenedFATPS(TransitionPathSampler):
    def get_trajectory_weight(self, trajectory):
        return self.kcm.s * self.kcm.activity(trajectory) - self.kcm.g * (np.sum(trajectory[0, :] + trajectory[-1, :]))

def draw_trajectory(trajectory, title, xlabel="Time Step", ylabel="Site", time_step=1.):
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(['w', 'r'])
    fig, ax= plt.subplots()
    ax.matshow(trajectory.T, cmap=cmap, aspect='auto')
    ax.set_title(title, pad=20)
    ax.set_xticklabels((np.arange(-1, len(trajectory)) * time_step))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.show()

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
