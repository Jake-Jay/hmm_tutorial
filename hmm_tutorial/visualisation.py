from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_transition_matrix(tm, labels):
    n_states = tm.shape[0]
    plt.imshow(tm)
    plt.xticks(np.arange(n_states), labels=labels)
    plt.yticks(np.arange(n_states), labels=labels)

    for i in range(n_states):
        for j in range(n_states):
            color = "w" if tm[i, j] < 0.6 else "k"
            text = plt.text(
                j, i, f"{tm[i, j]:.2f}", ha="center", va="center", color=color
            )
    plt.colorbar()
    plt.title("Transition probabilities")
    plt.tight_layout()
    plt.show()
