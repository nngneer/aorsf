"""
Utility functions for pyaorsf.
"""

import numpy as np
from typing import Optional


def concordance_index(
    time: np.ndarray,
    status: np.ndarray,
    risk: np.ndarray,
    tied_tol: float = 1e-8
) -> float:
    """
    Calculate Harrell's concordance index (C-index).

    The C-index measures the proportion of concordant pairs among all
    comparable pairs. A pair is concordant if the observation with the
    higher risk score has a shorter survival time.

    Parameters
    ----------
    time : ndarray of shape (n_samples,)
        Observed survival times.
    status : ndarray of shape (n_samples,)
        Event indicators (1 = event, 0 = censored).
    risk : ndarray of shape (n_samples,)
        Predicted risk scores (higher = higher risk).
    tied_tol : float, default=1e-8
        Tolerance for considering risk scores as tied.

    Returns
    -------
    c_index : float
        Concordance index between 0 and 1.
        - 1.0 = perfect concordance
        - 0.5 = random predictions
        - 0.0 = perfect discordance

    Examples
    --------
    >>> time = np.array([1, 2, 3, 4, 5])
    >>> status = np.array([1, 1, 0, 1, 1])
    >>> risk = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    >>> concordance_index(time, status, risk)
    1.0
    """
    time = np.asarray(time)
    status = np.asarray(status)
    risk = np.asarray(risk)

    n = len(time)
    concordant = 0
    discordant = 0
    tied_risk = 0

    for i in range(n):
        if status[i] == 0:
            continue  # Skip censored as the earlier event

        for j in range(n):
            if i == j:
                continue

            # Check if pair is comparable
            # i had event, j either had event later or was censored later
            if time[i] < time[j]:
                # Comparable pair: i had event before j
                risk_diff = risk[i] - risk[j]

                if risk_diff > tied_tol:
                    concordant += 1
                elif risk_diff < -tied_tol:
                    discordant += 1
                else:
                    tied_risk += 0.5

    total = concordant + discordant + tied_risk

    if total == 0:
        return 0.5  # No comparable pairs

    return (concordant + tied_risk) / total


def validate_survival_data(y: np.ndarray) -> tuple:
    """
    Validate and parse survival data.

    Parameters
    ----------
    y : ndarray
        Survival outcome data. Can be:
        - 2D array with columns [time, status]
        - Structured array with 'time' and 'status' fields

    Returns
    -------
    time : ndarray
        Survival times.
    status : ndarray
        Event indicators.

    Raises
    ------
    ValueError
        If y is not in a valid format.
    """
    y = np.asarray(y)

    # Check if structured array
    if y.dtype.names is not None:
        if 'time' not in y.dtype.names or 'status' not in y.dtype.names:
            raise ValueError(
                "Structured array must have 'time' and 'status' fields"
            )
        return y['time'], y['status']

    # Check if 2D array
    if y.ndim != 2:
        raise ValueError("y must be 2D array with columns [time, status]")

    if y.shape[1] != 2:
        raise ValueError("y must have exactly 2 columns: [time, status]")

    return y[:, 0], y[:, 1]


def check_random_state(seed: Optional[int]) -> np.random.RandomState:
    """
    Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : int or None
        If int, return a new RandomState with that seed.
        If None, return the global random state.

    Returns
    -------
    np.random.RandomState
        RandomState instance.
    """
    if seed is None:
        return np.random.mtrand._rand
    return np.random.RandomState(seed)
