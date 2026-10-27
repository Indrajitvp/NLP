"""Ex 5: HMM forward and backward procedures."""
import numpy as np

states = ["R", "S"]
symbols = ["A", "B"]
pi = np.array([1.0, 0.0])                       # initial probabilities
A = np.array([[0.7, 0.3],                       # transitions a_ij
              [0.4, 0.6]])
B = np.array([[0.8, 0.2],                       # emissions b_j(o)
              [0.3, 0.7]])
obs_seq = "A A B".split()
O = [symbols.index(o) for o in obs_seq]
N, T = len(states), len(O)


def forward(O):
    alpha = np.zeros((T, N))
    alpha[0] = pi * B[:, O[0]]                              # initialization
    for t in range(1, T):                                   # recursion
        alpha[t] = (alpha[t - 1] @ A) * B[:, O[t]]
    return alpha, alpha[-1].sum()                           # termination


def backward(O):
    beta = np.zeros((T, N))
    beta[-1] = 1.0                                          # initialization
    for t in range(T - 2, -1, -1):                          # recursion
        beta[t] = A @ (B[:, O[t + 1]] * beta[t + 1])
    return beta, (pi * B[:, O[0]] * beta[0]).sum()          # termination


alpha, p_fwd = forward(O)
beta, p_bwd = backward(O)
gamma = alpha * beta / p_fwd

print("Observation sequence:", obs_seq)
print("Alpha:\n", alpha)
print("Beta:\n", beta)
print("Gamma (P(state_t | O)):\n", gamma)
print(f"\nP(O | model) via forward  = {p_fwd:.6f}")
print(f"P(O | model) via backward = {p_bwd:.6f}")
assert np.isclose(p_fwd, p_bwd)
