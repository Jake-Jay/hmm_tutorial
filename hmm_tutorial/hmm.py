from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# --------
# Utils
# --------
def random_convex_combination(n: int) -> np.ndarray:
    """Return n coefficients which sum to one"""
    coefs = np.zeros(shape=n)
    coefs[0] = np.random.uniform(low=0., high=1.)
    for i in range(1, n-1):
        high = 1 - coefs.sum()
        coefs[i] = np.random.uniform(low=0, high=high)
    coefs[-1] = 1 - coefs.sum()
    np.random.shuffle(coefs)
    return coefs

def random_transition_matrix(n_states: int) -> np.ndarray:
    """Return a matrix where every row sums to one"""
    tm = np.zeros((n_states, n_states), dtype=np.float32)
    for i in range(n_states):
        tm[i, :] = random_convex_combination(n_states)
    return tm


# --------
# Classes
# --------
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

    def __init__(self, id: int, name: str, transition_probs: list[float], init_prob: float, mu: int) -> None:
        super().__init__(id, name, transition_probs, init_prob)

        self.mu = mu
        self.outputs = np.arange(30)
    
    def sample_output(self):
        return poisson.pmf()



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
                transition_probs=self.transition_matrix[i,:],
                init_prob=self.init_dist[i],
            )
            for i, name in enumerate(self.state_names)
        ]

    @property
    def n_states(self) -> int:
        return len(self.state_names)

    def plot_transition_matrix(self):
        plt.imshow(self.transition_matrix)
        plt.xticks(np.arange(self.n_states), labels=self.state_names)
        plt.yticks(np.arange(self.n_states), labels=self.state_names)

        for i in range(self.n_states):
            for j in range(self.n_states):
                color = "w" if self.transition_matrix[i, j] < 0.6 else "k"
                text = plt.text(
                    j, i, f"{self.transition_matrix[i, j]:.2f}",
                    ha="center", va="center", color=color
                )
        plt.colorbar()
        plt.title("Transition probabilities")
        plt.tight_layout()
        plt.show()

    def transition(self):
        transition_probs = self.transition_matrix[self.current_state_id,:]
        self.current_state_id = np.random.choice(self.state_ids, p=transition_probs)

    def trajectory(self, n_steps):
        t = [self.current_state_id]
        for i in range(n_steps):
            self.transition()
            t += [self.current_state_id]
            print(f"{t[i]}->{t[i+1]}")
        return [self.id2name[i] for i in t]
            

class HiddenMarkovModel(MarkovModel):

    def __init__(
        self,
        state_names: list[str],
        transition_matrix: np.ndarray,
        init_dist: list[float] | np.ndarray | None = None
    ) -> None:
        super().__init__(state_names, transition_matrix, init_dist)

