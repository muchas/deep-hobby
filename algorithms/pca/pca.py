from typing import Tuple

import numpy as np


def assert_column_vector(vector: np.array):
    assert len(vector.shape) == 2
    _, col_dim = vector.shape
    assert col_dim == 1


def cov(random_var1: np.array, random_var2: np.array, bias: bool = False) -> float:
    """
    cov(X, Y) = E[(X - E[X])(Y - E[Y])]
    """
    n = len(random_var1)
    mean1, mean2 = random_var1.mean(), random_var2.mean()
    deviations1 = random_var1 - mean1
    deviations2 = random_var2 - mean2
    denominator = n if bias else n - 1
    return np.dot(deviations1, deviations2) / denominator


def cov_matrix(variables: np.array, bias: bool = False):
    """
    Estimates the covariance matrix.

    Row values are the observations of a single random variable
    """
    _, n = variables.shape
    means = variables.mean(axis=1)
    deviations = variables - means.reshape(-1, 1)
    denominator = n if bias else n - 1
    return deviations @ deviations.T / denominator


def householder_reflector(hyperplane_vector: np.array) -> np.array:
    """
    Householder's reflector matrix

    Properties: orthogonal
    """
    assert_column_vector(hyperplane_vector)
    n = len(hyperplane_vector)
    return np.eye(n) - 2 * hyperplane_vector @ hyperplane_vector.T


def l2_norm(v: np.array) -> np.array:
    assert_column_vector(v)
    return np.sqrt(v.T.dot(v))


def householder_vector(v1: np.array, v2: np.array) -> np.array:
    """
    Vector which determines "reflection mirror" between v1 and v2.
    """
    diff = v1 - v2
    return diff / l2_norm(diff)


def qr_decomposition(matrix: np.array) -> Tuple[np.array, np.array]:
    """
    Householder reflection method

    The simplest, numerically stable QR decomp method.

    Be aware of the catastrophic cancellation when determining the Householder mirror.

    It's also useful for the linear least squares.
    """
    # TODO: think about faster HH transformation where we do not perform the full matrix-matrix multiplication
    assert len(matrix.shape) == 2
    m, n = matrix.shape
    assert n == m

    Q = np.eye(n)
    R = matrix.copy()

    for i in range(n):
        v_col = R[i:, i].reshape(-1, 1)

        v_e = np.zeros(len(v_col)).reshape(-1, 1)
        v_e[0] = l2_norm(v_col)

        # TODO: avoid catastrophic cancellation in the householder_vector
        # TODO: what about scaling factor to keep the values close to 1?
        # (this line helps to keep the same sign in the affected cell)
        v_e = -np.sign(v_col[0]) * v_e

        minor_reflector = householder_reflector(householder_vector(v_col, v_e))

        # Q: How do we ignore already "zeroed" columns?
        # A: We use "eye" matrix outside of the minor of interest
        reflector = np.eye(n)
        reflector[i:, i:] = minor_reflector

        R = reflector @ R
        Q = reflector @ Q

    return Q.T, R


def is_uppertriangular(matrix: np.array) -> bool:
    return np.allclose(matrix, np.triu(matrix))


def eigen_decomposition(matrix: np.array) -> Tuple[np.array, np.array]:
    """
    QR algorithm

    In general the eigenvalues of an upper-triangular matrix lie on its diagonal.
    """
    _, n = matrix.shape

    A = matrix.copy()
    V = np.eye(n)

    # TODO: improve convergence as "unshifted QR" do not always converge to the upper-triangular matrix

    while not is_uppertriangular(A):
        Q, R = qr_decomposition(A)
        A = R @ Q
        V = V @ Q

    return np.diag(A), V


def pca(data):
    """
    :param data:
    :return:
    """
    # normalize data
    mean, std = data.mean(axis=0), data.std(axis=0)
    data_normalized = (data - mean) / std  # N x D

    # calculate the covariance matrix
    # it tells you how much each feature "correlates with other features"
    # D x N @ N x D = D x D
    data_cov_matrix = cov_matrix(data_normalized.T)
    eigen_values, eigen_vectors = eigen_decomposition(data_cov_matrix)
    pass


def main():
    n = 100
    d = 30

    X = np.random.random((n, d))


if __name__ == '__main__':
    main()
