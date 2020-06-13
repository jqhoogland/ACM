 def no_step(self, state):
        """
        Carries out one step on ``state``.
        Iterate over all indices and probabilistically update (with probability
        ``self.prob_transition`` times the number of neighboring up spins)
        those spins whose neighbors have value 0

        """

        tmp_state = state.copy()

        # First, we flip all relevant parts
        for i in range(self.num_sites):


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


    def _step(self, state):
        # We choose a random state to update

        i = np.random.choice(np.arange(self.num_sites, dtype=np.int32))
        neighbors = self._get_neighbors(i, state)

        # The probability of switching to a particular state C' is
        # P(C -> C') = W(C -> C') / r(C),
        # where W(C -> C') is the rate of switching to that state C', and
        # where r(C) = sum_{C'} W(C -> C') is the sum of all transition rates

        total_rate = 0
        rate_swap = 0
        rate_flip = 0

        if state[i] == 0:
            # For an immobile state we are interested in the possibilitie of activation.

            rate_flip = self.get_rate_activation(neighbors)
            total_rate += rate_flip

            # There is also the possibility of swapping [0, 1] -> [1, 0]
            if i < self.num_sites - 1 and state[i + 1] == 1:
                rate_swap = self.rate_swap
                total_rate += rate_swap

        else:
            rate_flip = self.get_rate_inactivation(neighbors)
            total_rate += rate_flip

        prob_flip = rate_flip / total_rate
        prob_swap = rate_swap / total_rate

        choice = np.random.uniform()

        if choice < prob_flip:
            state[i] = 1 - state[i]
        elif choice < prob_flip + prob_swap:
            state[i], state[i + 1] = state[i + 1], state[i]

        return state
