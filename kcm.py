# ------------------------------------------------------------

# Code to implement Kinetically Constrained Models
#
# *References*
# [1] The Fredrickson-Aldersen Model [Frederickson and Alderson 1984](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.53.1244)
# [2] The East Model [Jackle and Eisinger 1991](https://link.springer.com/article/10.1007/BF01453764)
# [3] [Nicholas B. Tutto 2018](https://tps.phys.tue.nl/lm-live/fa-ising/)

# ------------------------------------------------------------

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
     - Spin ``i`` can flip with probability ``prob_transition`` iff spin ``i+1``
       (to the right or left, i.e. "east") is "up"
     - Boundaries are closed.

    For convenience sake, we interpret the spins as occupation numbers $$0, 1$$
    rather than the conventional up/down $$\pm 1$$.

    This class produces a single trajectory.

    The rate for flipping a spin 0 -> 1 is ``= weight_activation * C_i``
    The rate for flipping a spin 1 -> 0 is ``= weight_activation * inactivation_ratio * C_i``
    The rate of swapping (0, 1) -> (1,0) is ``=rate_activation * rate_swap``

    """

    def __init__(self, rate_swap=None, biasing_field=None, temperature=1., coupling_energy=0., inactivation_ratio=None, softening_param=0., num_sites=50, num_steps=100, num_burnin_steps=0):
        """
        :param float rate_swap: The swapping rate for the transition (0, 1) -> (1, 0)
        :param float temperature: The temperature which controls the transition rate 1 -> 0
        :param float coupling_energy: The energy which together with the temperature controls the transition rate 1 -> 0
        :param float softening_param: The extent to which even immobile regions are allowed to become mobile
        :Param int num_sites: The number of sites to include in the chain.
        :param int num_steps: The number of steps to include in a trajectory.
        :param int num_burnin_steps: The number of steps to equilibrate a
            randomly initialized configuration.

        """
        super(SoftenedFA, self).__init__(0., num_sites, num_steps, num_burnin_steps)
        self.temperature = temperature
        self.softening_param = softening_param
        self.coupling_energy  = coupling_energy

        # Derived quanities
        if inactivation_ratio is None:
            inactivation_ratio = np.exp(-coupling_energy / temperature) # Defaults to 1. i.e. 0 -> 1 is as likely as 1 -> 0

        self.inactivation_ratio = inactivation_ratio

        # Rates
        self.get_neighbor_constraint = lambda neighbors: (np.sum(neighbors) + np.size(neighbors) * self.softening_param / 2)
        self.get_rate_activation = lambda neighbors:  self.get_neighbor_constraint(neighbors)
        self.get_rate_inactivation = lambda neighbors:  (self.inactivation_ratio * self.get_neighbor_constraint(neighbors))

        # Probabilities
        self.get_prob_from_rate = lambda rate: 1 - np.exp(-rate)

        self.get_prob_activation = lambda neighbors: self.get_prob_from_rate(self.get_rate_activation(neighbors))
        self.get_prob_inactivation = lambda neighbors: self.get_prob_from_rate(self.get_rate_inactivation(neighbors))

        print("Neighboring spins; Probability activation; Probability inactivation")
        for neighbors in [np.array([1, 1]), np.array([1,0 ]), np.array([0, 1]), np.array([0, 0])]:
            print(neighbors, self.get_prob_activation(neighbors), self.get_prob_inactivation(neighbors))

        self.critical_biasing_field = self.get_critical_biasing_field()

        print("Critical biasing field: {}".format(self.critical_biasing_field))

        if biasing_field is None:
            biasing_field = self.critical_biasing_field

        self.biasing_field = self.critical_biasing_field

        if rate_swap:
            self.rate_swap = rate_swap
        else:
            self.rate_swap = self.get_rate_swap(biasing_field)

        self.biasing_field = biasing_field

        self.prob_swap = self.get_prob_from_rate(self.rate_swap)
        print("Probability swap: {}".format(self.prob_swap))

    def get_rate_swap(self, s):
        return (1. - self.inactivation_ratio + np.sqrt((1. - self.inactivation_ratio) ** 2. + 4. * np.exp(-s) * self.inactivation_ratio)) / 2.

    def get_critical_biasing_field(self):
        D = self.get_rate_swap
        lhs = lambda s: (1. - self.inactivation_ratio) / (1. + self.softening_param)
        rhs = lambda s: np.sqrt((1. - self.inactivation_ratio - D(s) * (1 - np.exp(-s))) ** 2. +4. * np.exp(-2. * s) * self.inactivation_ratio) - (1. - np.exp(-s) * D(s))

        sol = optimize.root_scalar(lambda s: rhs(s) - lhs(s), x0=0, x1=.1)

        return sol.root

    @staticmethod
    def _get_neighbors(index, state):
        neighbors = []

        if index == 0:
            neighbors = [state[index + 1]]
        elif index == len(state) - 1:
            neighbors = [state[index - 1]]
        else:
            neighbors = [state[index - 1], state[index + 1]]

        return np.array(neighbors)

    def _step(self, state):
        """
        Carries out one step on ``state``.
        Iterate over all indices and probabilistically update (with probability
        ``self.prob_transition`` times the number of neighboring up spins)
        those spins whose neighbors have value 0

        """

        tmp_state = state.copy()

        # First, we flip all relevant parts
        for i in range(self.num_sites):
            neighbors = self._get_neighbors(i, state)

            # Slow state
            if state[i] == 0 and  np.random.uniform() < self.get_prob_activation(neighbors):
                tmp_state[i] = 1

            # Fast state inactivation
            elif state[i] == 1 and np.random.uniform() < self.get_prob_inactivation(neighbors):
                tmp_state[i] = 0

        # Swap afterwards according to the values of the previous condition
        i = 0

        while i < self.num_sites - 1:
            if state[i] == 0 and state[i + 1] == 1 and np.random.uniform() < self.prob_swap:
                # I actually think it might not matter whether you flip based on state or tmp_state
                tmp_state[i], tmp_state[i + 1] == tmp_state[i + 1], tmp_state[i]
                i += 2 # Same here.
            else:
                i += 1

        return tmp_state

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

        for index in range(self.num_sites):

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
                direction = 1

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

    cmap = ListedColormap(['w', 'r'])
    fig = plt.figure(figsize=(10, 10))
    plt.matshow(trajectory.T, cmap=cmap, fignum=fig.number)
    plt.ylabel("Site")
    plt.xlabel("Time step")
    plt.title("Trajectory")

    plt.show()


if __name__ == "__main__":
    fa_kcm = OneSpinFAKCM(0.5, 0.5, 25, 200, 0.1, 0.1, 1, 50000)
    trajectory = fa_kcm.gen_trajectory()
    draw_trajectory(trajectory)
