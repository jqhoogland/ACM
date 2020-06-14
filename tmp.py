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


def step2(self, state):
            # STEP 1. DETERMINE WEIGHTS W(C -> C')

        # We determine the weight W(C -> C') for either swapping or flipping at each site.
        # (if a swap is not allowed, e.g. a pair [0, 0] or [1, 1], then it gets weight 0)
        swap_weights = np.zeros(self.num_sites)
        flip_weights = np.zeros(self.num_sites)

        for i in range(self.num_sites):
            neighbors = self._get_neighbors(i, state)

            if state[i] == 0: # The state is inactive
                flip_weights[i] = self.get_rate_activation(neighbors)

                # There is also the possibility of swapping [0, 1] -> [1, 0]
                if state[(i + 1) % self.num_sites] == 1:
                    swap_weights[i] = self.rate_swap

            else: # The state is active
                flip_weights[i] = self.get_rate_inactivation(neighbors)

                # There is also the possibility of swapping [1, 0] -> [0, 1]
                # In Elmatad et al., they only discuss [0, 1] -> [0, 1], but
                # we implicitly need the inverse in order to satisfy detailed balance
                if state[(i + 1) % self.num_sites] == 0:
                    swap_weights[i] = self.rate_swap

        total_weight = np.sum(swap_weights) + np.sum(flip_weights)

        # STEP 2. DETERMINE WHETHER TO UPDATE C
        # The characteristic time of C -> C' is equal to r(C), the sum over all W(C -> C')
        # We update with a probability 1 - exp(-r(C))

        time_step = 0.001
        if np.random.uniform() < np.exp(-total_weight * time_step):
             return state

        # STEP 3. CHOOSE A C'
        # We randomly choose a C' by its normalized weight (probability).

        swap_probs = swap_weights / total_weight
        flip_probs = flip_weights / total_weight

        choice = np.random.uniform()
        prob = flip_probs[0]

        logging.debug("Choice: {}".format(choice))
        logging.debug("Flips: {}".format(flip_probs))
        logging.debug("Swaps: {}".format(swap_probs))

        # We choose to flip first index i so that the cumulative probability of flips for
        # all sites up to and including index i exceeds the randomly generated number.
        # If the cumulative probabilities don't reach this point, we go on to considering swaps.
        # We choose to flip the first index j so that the cumulative probability of swaps
        # on all sites up to and including index j PLUS the probability of all flips exceeds the randomly generated number

        i = 0
        while prob < choice and i < self.num_sites - 1:
            i += 1
            prob += flip_probs[i]

        if choice < prob:
            state = self._flip(state, i)
        else:
            j = 0
            prob += swap_probs[0]
            while prob < choice and j < self.num_sites - 1:
                j += 1
                prob += swap_probs[j]

            state = self._swap(state, j)

        return state
