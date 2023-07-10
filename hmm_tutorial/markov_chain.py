from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from hmm_tutorial.utils import random_convex_combination


class State:
    def __init__(
        self,
        id: int,
        name: str,
        transition_probs: list[float],
        init_prob: float,
    ) -> None:
        self.id = id
        self.name = name
        self.transition_probs = transition_probs
        self.init_prob = init_prob

    def __repr__(self) -> str:
        return f"State {self.id}: {self.name}"


class PoissonState(State):
    def __init__(
        self,
        id: int,
        name: str,
        transition_probs: list[float],
        init_prob: float,
        mu: int,
    ) -> None:
        super().__init__(id, name, transition_probs, init_prob)

        self.mu = mu

    def sample_output(self):
        return poisson.rvs(mu=self.mu)

    def output_probability(self, output: int) -> float:
        return poisson.pmf(k=output, mu=self.mu)


class MarkovModel:
    def __init__(
        self,
        state_names: list[str],
        transition_matrix: np.ndarray,
        init_dist: list[float] | np.ndarray | None = None,
    ) -> None:
        self.state_names = state_names
        self.transition_matrix = transition_matrix

        if init_dist is None:
            self.init_dist = random_convex_combination(n=len(state_names))
        else:
            self.init_dist = init_dist

        self._instantiate_states()
        self.state_ids = [s.id for s in self.states]
        self.id2name = {s.id: s.name for s in self.states}
        self.name2id = {s.name: s.id for s in self.states}

        self.start_state_id = np.random.choice(self.state_ids, p=self.init_dist)
        self.current_state_id = self.start_state_id

    def _instantiate_states(self) -> None:
        self.states = [
            State(
                id=i,
                name=name,
                transition_probs=self.transition_matrix[i, :],
                init_prob=self.init_dist[i],
            )
            for i, name in enumerate(self.state_names)
        ]

    @property
    def n_states(self) -> int:
        return len(self.state_names)

    def transition(self):
        transition_probs = self.transition_matrix[self.current_state_id, :]
        self.current_state_id = np.random.choice(self.state_ids, p=transition_probs)

    def trajectory(self, n_steps):
        t = [self.current_state_id]
        for i in range(n_steps):
            self.transition()
            t += [self.current_state_id]
            print(f"{t[i]}->{t[i+1]}")
        return [self.id2name[i] for i in t]
