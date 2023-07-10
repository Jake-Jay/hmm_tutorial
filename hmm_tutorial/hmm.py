from __future__ import annotations

import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

from hmm_tutorial.markov_chain import MarkovModel
from hmm_tutorial.markov_chain import PoissonState


class PoissonHiddenMarkovModel(MarkovModel):
    """HMM where observations are parameterised by a Poisson distribution"""

    def __init__(
        self,
        state_names: list[str],
        lambdas: list[int],
        transition_matrix: np.ndarray,
        init_dist: list[float] | np.ndarray | None = None,
    ) -> None:
        self.lambdas = lambdas
        super().__init__(state_names, transition_matrix, init_dist)

    def decode(
        self, observations: list[int], return_labels: bool = False
    ) -> tuple[float, list[str | int]]:
        """Given a set of observations, find the most likely sequence of state

        Args:
            observations (list[int]): Sequence of numerical observations

        Returns:
            tuple[float, list[str]]: (best path's probability, best path)
        """

        path_prob, path = self._viterbi(observations)
        if return_labels:
            path = [self.id2name[state_id] for state_id in path]
        return path_prob, path

    def likelihood(self, observations: list[int], naive: bool = False) -> float:
        """Likelihood of a sequence of observations given an HMM"""

        if naive:
            return self._forward_naive(observations)
        else:
            return self._forward(observations)

    def trajectory(self, n_steps) -> tuple[list[str], list[int]]:
        """A random walk"""
        state_trajectory = [self.current_state_id]
        observations = [self.states[self.current_state_id].sample_output()]
        for i in range(n_steps):
            self.transition()
            state_trajectory += [self.current_state_id]
            observations += [self.states[self.current_state_id].sample_output()]
        return [self.id2name[i] for i in state_trajectory], observations

    def _instantiate_states(self) -> None:
        """Create state objects with specific output distributions"""
        self.states = [
            PoissonState(
                id=i,
                name=name,
                transition_probs=self.transition_matrix[i, :],
                init_prob=self.init_dist[i],
                mu=self.lambdas[i],
            )
            for i, name in enumerate(self.state_names)
        ]

    def _forward(self, observations: list[int]) -> float:
        n_steps = len(observations)
        n_states = self.n_states

        # initialise
        probability_matrix = np.zeros((n_states, n_steps))
        for state_id in range(n_states):
            state = self.states[state_id]
            probability_matrix[state_id, 0] = self.init_dist[
                state_id
            ] * state.output_probability(observations[0])

        # dynamic programming step
        for step in range(1, n_steps):
            for current_state_id in range(n_states):
                current_state = self.states[current_state_id]
                for prev_state_id in range(n_states):
                    probability_matrix[current_state_id, step] += (
                        probability_matrix[prev_state_id, step - 1]
                        * self.transition_matrix[prev_state_id, current_state_id]
                        * current_state.output_probability(observations[step])
                    )

        # sum final column
        return probability_matrix[:, -1].sum()

    def _forward_naive(self, observations: list[int]) -> float:
        """Determine the likelihood of a sequence without using dynamic programming."""

        def _get_all_trajectories(n_steps) -> list[tuple]:
            """All possible combinations of trajectory"""
            return list(itertools.product(self.state_ids, repeat=n_steps))

        def _probability_trajectory(trajectory) -> float:
            """P(Q) -> Find the probability of a single trajectory"""
            p = self.init_dist[trajectory[0]]
            for step in range(len(trajectory) - 1):
                current_state = trajectory[step]
                next_state = trajectory[step + 1]
                p *= self.transition_matrix[current_state, next_state]
            return p

        def _probability_observations_given_trajectory(
            observations: list[int], trajectory: list[int]
        ) -> float:
            """P(O|Q) -> conditional probability of observations given the trajectory"""
            p_obs_given_state = 1
            for state_id, observation in zip(trajectory, observations):
                current_state: PoissonState = self.states[state_id]
                p_obs_given_state *= current_state.output_probability(observation)

            return p_obs_given_state

        def _probability_observation_and_trajectory(
            observations: list[int], trajectory: list[int]
        ) -> float:
            """P(O, Q) = P(O | Q) x P(Q) -> joint probability of a sequence of observations and states"""
            p_obs_given_state = _probability_observations_given_trajectory(
                observations, trajectory
            )
            p_trajectory = _probability_trajectory(trajectory)
            return p_obs_given_state * p_trajectory

        n_steps = len(observations)
        assert n_steps < 15, "we do not want to let memory explode"
        trajectories = _get_all_trajectories(n_steps)

        likelihood = 0
        # sum_{Q} P(O | Q) x P(Q)
        for trajectory in trajectories:
            likelihood += (
                # P(O, Q) = P(O | Q) x P(Q)
                _probability_observation_and_trajectory(observations, trajectory)
            )
        return likelihood

    def _viterbi(self, observations: list[int]) -> tuple[list[str], float]:
        """Run viterbi to determine the best sequence of states given some observations

        Args:
            observations (list[int]): Sequence of numerical observations

        Returns:
            tuple[float, list[str]]: (best path's probability, best path)
        """
        n_steps = len(observations)
        n_states = self.n_states

        # initialise
        v = np.zeros((n_states, n_steps))  # stores viterbi products
        path_trace = np.zeros_like(
            v, dtype=int
        )  # stores a pointer to the best previous state from each state
        for state_id in range(n_states):
            state = self.states[state_id]
            v[state_id, 0] = self.init_dist[state_id] * state.output_probability(
                observations[0]
            )
            path_trace[state_id, 0] = -1

        # dynamic programming step
        for step in range(1, n_steps):
            for current_state_id in range(n_states):
                current_state = self.states[current_state_id]
                path_probabilities = [
                    (
                        v[prev_state_id, step - 1]
                        * self.transition_matrix[prev_state_id, current_state_id]
                        * current_state.output_probability(observations[step])
                    )
                    for prev_state_id in range(n_states)
                ]
                v[current_state_id, step] = np.max(path_probabilities)
                path_trace[current_state_id, step] = np.argmax(path_probabilities)

        # find the best path by tracing back through stored indices
        best_final_state = np.argmax(v[:, -1])
        best_path = [best_final_state]
        best_previous_step = path_trace[best_final_state, -1]
        best_path += [best_previous_step]
        for i in range(2, n_steps):
            best_previous_step = path_trace[best_previous_step, -i]
            best_path += [best_previous_step]
        best_path.reverse()

        return np.max(v[:-1]), best_path
