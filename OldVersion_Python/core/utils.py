"""Utility functions.

Translated from: x_in_s.m
"""
import numpy as np


def x_in_s(S, x):
    """Find the index of solution x in set S.

    Translated from x_in_s.m:
      Find the first appearance of column x in matrix S.

    Parameters
    ----------
    S : list of np.ndarray or np.ndarray of shape (K, d)
        Set of solutions. Each row is a solution.
    x : np.ndarray of shape (d,)
        Solution to find.

    Returns
    -------
    int or None
        Index of x in S, or None if not found.
    """
    if S is None or len(S) == 0:
        return None
    x = np.asarray(x)
    for k in range(len(S)):
        if np.array_equal(S[k], x):
            return k
    return None


def batch_x_in_s(S, X):
    """Find indices of each row in X within set S (vectorized).

    Parameters
    ----------
    S : list of np.ndarray
        Set of solutions.
    X : list of np.ndarray or np.ndarray of shape (K, d)
        Solutions to look up.

    Returns
    -------
    list of (int or None)
        Index of each X[k] in S, or None if not found.
    """
    if S is None or len(S) == 0:
        return [None] * len(X)
    S_arr = np.array(S)
    X_arr = np.array(X) if not isinstance(X, np.ndarray) else X
    results = [None] * len(X_arr)
    for k in range(len(X_arr)):
        # Vectorized comparison
        matches = np.all(S_arr == X_arr[k], axis=1)
        idx = np.where(matches)[0]
        if len(idx) > 0:
            results[k] = int(idx[0])
    return results
