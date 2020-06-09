# ------------------------------------------------------------

# Code to implement Kinetically Constrained Models
#
# *References*
# [1] The Fredrickson-Aldersen Model [Frederickson and Alderson 1984](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.53.1244)
# [2] The East Model [Jackle and Eisinger 1991](https://link.springer.com/article/10.1007/BF01453764)

# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

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
        self.prob_transition = prob_transition
        self.num_sites = num_sites
        self.num_steps = num_steps
        self.num_burnin_steps = num_burnin_steps

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
        ``self.prob_transition`` times the number of neighboring up spins)
        those spins whose right-neighbors have value 0

        """

        for i in range(self.num_sites - 1):
            if state[i + 1] == 1 and np.random.uniform() < self.prob_transition:
                state[i] = 1 - state[i]

        return state



def draw_trajectory(trajectory):
    """
    :param (np.array) trajectory: A trajectory of shape
        (self.num_steps, self.num_sites). The rows are states.
    """

    plt.matshow(trajectory.T)
    plt.ylabel("Site")
    plt.xlabel("Time step")
    plt.title("Trajectory")
    plt.show()
