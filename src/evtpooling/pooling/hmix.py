import copy
import time
from functools import cache

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import norm

from evtpooling.constants import DEBUG


# Define a custom exception for specific errors if certain
# conditions are not met defined below
class ExtractError(Exception):
    """Custom exception for errors during data extraction."""

    pass


def state_probability(
    x: float, i: int, theta: tuple[list[float], list[tuple[float, float]]]
) -> float:
    """
    Calculate the state probability for a given value x and state index i
    using the parameters from a Gaussian Mixture Model (GMM).

    Parameters:
    ----------
    x : float
        The value for which the state probability is calculated.
    i : int
        The index of the state for which the probability is calculated.
    theta : tuple[list[float], list[tuple[float, float]]]
        A tuple containing:
        - pi_list: A list of mixing coefficients for each state.
        - phi_list: A list of tuples, each containing the mean and standard deviation
          for the corresponding state.

    Returns:
    -------
    float
        The state probability for the given value x and state index i.
        Returns 0.0 if the denominator is less than 0.
    """

    pi_list, phi_list = theta

    if sum(pi_list) != 1.0:
        raise ValueError("Mixing coefficients (pi_list) must sum to 1.")

    # Validate inputs
    if not (len(pi_list) == len(phi_list)):
        raise ValueError("Mismatch between number of mixture weights and components.")

    mu_i, sigma_i = phi_list[i]

    numerator = pi_list[i] * norm.pdf(x, loc=mu_i, scale=sigma_i)
    denominator = sum(
        pi * norm.pdf(x, loc=mu, scale=sigma)
        for pi, (mu, sigma) in zip(pi_list, phi_list, strict=False)
    )

    if DEBUG:
        print(
            f"State probability for component {i} at x={x:.5f}:"
            f"numerator={numerator:.5f}, denominator={denominator:.5f}"
        )

    if np.isclose(numerator, 0.0):
        return 0.0

    if np.isclose(denominator, 0.0, atol=1e-16):
        raise ZeroDivisionError(
            "Can't return the state probability as the denominator is (close to zero)"
        )

    return numerator / denominator


@cache
def cached_state_probability(x: float, i: int, pi_tuple: tuple, phi_tuple: tuple) -> float:
    """
    Memoized version of state_probability using hashable tuples for caching.
    """
    pi_list = list(pi_tuple)
    phi_list = list(phi_tuple)

    return state_probability(x, i, (pi_list, phi_list))


def precompute_pdf_matrix(alpha_list, phi_list):
    """
    Precompute norm.pdf(x_j, mu_i, sigma_i) for all x_j in alpha_list and i in phi_list
    Returns a matrix [len(alpha_list) x len(phi_list)]
    """
    pdf_matrix = []
    for x in alpha_list:
        row = [norm.pdf(x, loc=mu, scale=sigma) for (mu, sigma) in phi_list]
        pdf_matrix.append(row)
    return pdf_matrix


def non_parametric_density(
    x: float, alpha_list: list[float], theta: tuple[list[float], list[tuple[float, float]]]
) -> float:
    """
    Compute the non-parametric density estimate at a point x using a Gaussian Mixture Model (GMM).
    This function uses the Epanechnikov kernel for density estimation.

    Parameters:
    ----------
    x : float
        The point at which to evaluate the density estimate.
    alpha_list : list[float]
        A list of values at which the density is estimated.
    theta : tuple[list[float], list[tuple[float, float]]]
        A tuple containing:
        - pi_list: A list of mixing coefficients for each state.
        - phi_list: A list of tuples, each containing the mean and standard deviation
                    for the corresponding state.

    Returns:
    -------
    float
        The estimated density at point x. If the input list is empty, raises ZeroDivisionError.
    Raises:
    ------
    ZeroDivisionError
        If the input list is empty, indicating that no density can be computed.
    """

    def epanechnikov_kernel(x: float) -> float:
        return 0.75 * (1 - x**2) if abs(x) <= 1 else 0.0

    def compute_bandwidths(n, sigma_list):
        return [2.283 * n ** (-0.287) * sigma if sigma > 0 else 0.0 for sigma in sigma_list]

    pi_list, phi_list = theta

    # Validate inputs
    if not (len(pi_list) == len(phi_list)):
        raise ValueError("Mismatch between number of mixture weights and components.")

    k = len(pi_list)
    sigma_list = [sigma for (_, sigma) in phi_list]
    bandwidths = compute_bandwidths(len(alpha_list), sigma_list)

    total = 0.0
    for j in range(len(alpha_list)):
        x_j = alpha_list[j]
        for i in range(k):
            aij = cached_state_probability(x_j, i, tuple(pi_list), tuple(map(tuple, phi_list)))
            cni = bandwidths[i]
            u = (x - x_j) / cni

            if DEBUG:
                print(f"aij={aij:.5f}, cni={cni:.5f}, u={u:.5f} for x_j={x_j:.5f} and x={x}, i={i}")

            total += (aij / cni) * epanechnikov_kernel(u)

    try:
        return total / len(alpha_list)
    except ZeroDivisionError as e:
        raise ZeroDivisionError("The input list is empty, cannot compute density.") from e


def integrand_i(
    x: float, i: int, alpha_list: list[float], theta: tuple[list[float], list[tuple[float, float]]]
) -> float:
    """
    Compute the integrand for the Gaussian Mixture Model (GMM) density function.

    Parameters:
    ----------
    x : float
        The value at which to evaluate the integrand.
    i : int
        The index of the state for which the integrand is computed.
    theta : tuple[list[float], list[tuple[float, float]]]
        A tuple containing:
        - pi_list: A list of mixing coefficients for each state.
        - phi_list: A list of tuples, each containing the mean and standard deviation
          for the corresponding state.

    Returns:
    -------
    float
        The value of the integrand at x for state i.
    """
    pi_list, phi_list = theta

    # Validate inputs
    if not (len(pi_list) == len(phi_list)):
        raise ValueError("Mismatch between number of mixture weights and components.")

    mu_i, sigma_i = phi_list[i]

    try:
        ai = state_probability(x, i, theta)
        fx_phi = norm.pdf(x, loc=mu_i, scale=sigma_i)
        ghat = non_parametric_density(x, alpha_list, theta)
        val = np.sqrt(ai * fx_phi * ghat)
        # print(f"Integrand for component {i} at x={x:.5f}: ai={ai:.5f},"
        #       f"fx_phi={fx_phi:.5f}, ghat={ghat}, val={val:.5f}")

        return val
    except ExtractError as e:
        raise ExtractError(f"Error in computing integrand: {e}") from e


def h_total_int(
    alpha_list: list[float],
    i: int,
    theta: tuple[list[float], list[tuple[float, float]]],
    bounds: tuple[float, float] = (1.0, 10.0),
) -> float:
    """ """
    # Validate inputs
    if not alpha_list or any(a <= 0 for a in alpha_list):
        raise ValueError("alpha_list must be non-empty and contain only positive values.")

    lower, higher = bounds

    try:
        start = time.time()
        integral_i, _ = quad(
            integrand_i,
            lower,
            higher,
            args=(i, alpha_list, theta),
            limit=200,
        )
        print(f"Integration took {time.time() - start:.2f} seconds")
        if not np.isfinite(integral_i):
            raise ArithmeticError(f"Non-finite integral for component {i}: {integral_i}")
        return integral_i

    except Exception as e:
        raise RuntimeError(f"Failed to compute integral for component {i}: {e}") from e


def h_theta(alpha_list: list[float], theta: tuple[list[float], list[tuple[float, float]]]) -> float:
    """
    Compute the value of h(theta) for the Gaussian Mixture Model (GMM).

    Parameters:
    ----------
    alpha_list : list[float]
        A list of values at which the density is estimated.
    theta : tuple[list[float], list[tuple[float, float]]]
        A tuple containing:
        - pi_list: A list of mixing coefficients for each state.
        - phi_list: A list of tuples, each containing the mean and standard deviation
          for the corresponding state.

    Returns:
    -------
    float
        The value of h(theta), which is the integral of the GMM density function
        over the specified range, weighted by the mixing coefficients.
    """
    pi_list, phi_list = theta

    # Validate inputs
    if not (len(pi_list) == len(phi_list)):
        raise ValueError("Mismatch between number of mixture weights and components.")
    if not alpha_list or any(a <= 0 for a in alpha_list):
        raise ValueError("alpha_list must be non-empty and contain only positive values.")

    k = len(pi_list)
    h_total = 0.0

    for i in range(k):
        try:
            integral_i = h_total_int(alpha_list, i, theta)
            weighted_term = np.sqrt(pi_list[i]) * integral_i
            if not np.isfinite(weighted_term):
                raise ArithmeticError(f"Non-finite integral for component {i}: {weighted_term}")
            h_total += weighted_term

        except Exception as e:
            raise RuntimeError(f"Failed to compute integral for component {i}: {e}") from e

    return h_total


def optimize_phi_i(
    i: int,
    alpha_list: list[float],
    theta: tuple[list[float], list[tuple[float, float]]],
    bounds_arg: tuple[float, float] = (1.0, 10.0),
) -> tuple[float, float]:
    """
    Optimize the parameters (mu, sigma) for the i-th component of the Gaussian Mixture Model (GMM)
    using the h_theta function.

    Parameters:
    ----------
    i : int
        The index of the component to optimize.
    alpha_list : list[float]
        A list of values at which the density is estimated.
    theta : tuple[list[float], list[tuple[float, float]]]
        A tuple containing:
        - pi_list: A list of mixing coefficients for each state.
        - phi_list: A list of tuples, each containing the mean and standard deviation
            for the corresponding state.

    Returns:
    -------
    tuple[float, float]
        The optimized parameters (mu_i*, sigma_i*) for the i-th component.

    Raises:
    ------
    ValueError
        If the number of mixture weights does not match the number of components,
        or if alpha_list is empty or contains non-positive values.
    RuntimeError
        If the optimization fails for the i-th component.
    """
    pi_list, phi_list = theta

    # Validate inputs
    if not (len(pi_list) == len(phi_list)):
        raise ValueError("Mismatch between number of mixture weights and components.")
    if not alpha_list or any(a <= 0 for a in alpha_list):
        raise ValueError("alpha_list must be non-empty and contain only positive values.")

    def objective(phi):
        mu, sigma = phi
        if sigma <= 0:
            return np.inf
        new_phi_list = copy.deepcopy(phi_list)
        new_phi_list[i] = (mu, sigma)
        new_theta = (pi_list, new_phi_list)
        return -h_total_int(alpha_list=alpha_list, i=i, theta=new_theta, bounds=bounds_arg)

    # Initial guess
    mu0, sigma0 = phi_list[i]
    result = minimize(objective, x0=[mu0, sigma0], bounds=[bounds_arg, (0.05, 5)])

    if not result.success:
        raise RuntimeError(f"Optimization failed for component {i}: {result.message}")

    return result.x  # returns (mu_i*, sigma_i*)


def optimize_pi_i(
    i: int,
    alpha_list: list[float],
    theta: tuple[list[float, list[tuple[float, float]]]],
    bounds: tuple[float, float] = (1.0, 10.0),
) -> float:
    """
    Optimize the mixing coefficient pi_i for the i-th component of the Gaussian Mixture Model (GMM)
    using the h_total_int function.

    Parameters:
    ----------
    i : int
        The index of the component for which to optimize the mixing coefficient.
    alpha_list : list[float]
        A list of values at which the density is estimated.
    theta : tuple[list[float], list[tuple[float, float]]]
        A tuple containing:
        - pi_list: A list of mixing coefficients for each state.
        - phi_list: A list of tuples, each containing the mean and standard deviation
            for the corresponding state.
    bounds : tuple[float, float], optional
        The bounds for the optimization of pi_i. Default is (1.0, 10.0).

    Returns:
    -------
    float
        The optimized mixing coefficient pi_i* for the i-th component.
    """
    pi_list, phi_list = theta

    # Validate inputs
    if not (len(pi_list) == len(phi_list)):
        raise ValueError("Mismatch between number of mixture weights and components.")
    if not alpha_list or any(a <= 0 for a in alpha_list):
        raise ValueError("alpha_list must be non-empty and contain only positive values.")

    nominator = h_total_int(alpha_list, i, theta, bounds) ** 2
    denominator = sum(h_total_int(alpha_list, j, theta, bounds) ** 2 for j in range(len(theta[0])))

    try:
        return nominator / denominator
    except ZeroDivisionError as e:
        raise ZeroDivisionError("The denominator is zero, cannot compute pi_i.") from e


def hmix_func(
    alpha_list: list[float],
    num_clusters: int,
    theta: tuple[list[float], list[tuple[float, float]]] = None,
    threshold: float = 1e-6,
    max_iter: int = 1000,
) -> tuple[list[float], list[tuple[float, float]]]:
    """
    Perform hierarchical mixture optimization on the given parameters.

    Parameters:
    ----------
    alpha_list : list[float]
        A list of values at which the density is estimated.
    num_clusters : int
        The number of clusters to optimize.
    theta : tuple[list[float], list[tuple[float, float]]]
        A tuple containing:
        - pi_list: A list of mixing coefficients for each state.
        - phi_list: A list of tuples, each containing the mean and standard deviation
          for the corresponding state.
    threshold : float, optional
        The convergence threshold for the optimization. Default is 1e-6.
    max_iter : int, optional
        The maximum number of iterations for the optimization. Default is 1000.

    Returns:
    -------
    tuple[list[float], list[tuple[float, float]]]
        The optimized parameters (pi_list, phi_list) after hierarchical mixture optimization.
    """

    if theta is None:
        # Initialize theta with equal weights and random means and standard deviations
        pi_list = [1.0 / num_clusters] * num_clusters
        phi_list = [
            (np.random.uniform(0, 10), np.random.uniform(0.1, 2)) for _ in range(num_clusters)
        ]
        theta = (pi_list, phi_list)
    else:
        # Ensure theta is a tuple of lists
        if (
            not isinstance(theta, tuple)
            or len(theta) != 2
            or not all(isinstance(lst, list) for lst in theta)
            or num_clusters != len(theta[0])
        ):
            raise ValueError(
                "theta must be a tuple of two lists with the same length as num_clusters."
            )

    pi_list, phi_list = theta

    # Validate inputs
    if not (len(pi_list) == len(phi_list)):
        raise ValueError("Mismatch between number of mixture weights and components.")
    if not alpha_list or any(a <= 0 for a in alpha_list):
        raise ValueError("alpha_list must be non-empty and contain only positive values.")

    h_theta_t = 100.0
    h_theta_t1 = 0.0
    t = 0

    while abs(h_theta_t - h_theta_t1) > threshold and t < max_iter:
        for i in range(num_clusters):
            # Optimize the parameters for each component
            if DEBUG:
                print(
                    f"Optimizing component {i} with initial parameters: "
                    f"mu={phi_list[i][0]:.3f}, sigma={phi_list[i][1]:.3f}"
                )
            mu_opt, sigma_opt = optimize_phi_i(
                i, alpha_list, theta, bounds_arg=(1.0, max(alpha_list) + 2.0)
            )
            pi_opt = optimize_pi_i(i, alpha_list, theta)
            if DEBUG:
                print(
                    f"Optimization result for component {i}: mu_opt={mu_opt:.3f}, "
                    f"sigma_opt={sigma_opt:.3f}, pi={pi_opt:.3f}"
                )
                print("-" * 70)

            # Update the parameters
            pi_list[i] = pi_opt
            phi_list[i] = (mu_opt, sigma_opt)

        total = sum(pi_list)
        if not np.isclose(total, 1.0):
            print(f"Warning: Mixing coefficients do not sum to 1. Normalizing: {total:.3f}")
            pi_list = [pi / total for pi in pi_list]

        h_theta_t1 = h_theta_t
        h_theta_t = h_theta(alpha_list, (pi_list, phi_list))
        t += 1
        if DEBUG:
            print(f"Iteration {t}: h(theta) = {h_theta_t:.6f}, h(theta_t1) = {h_theta_t1:.6f}")

        if t == max_iter:
            print(f"Warning: Maximum iterations {max_iter} reached without convergence.")
            print(f"Final parameters: pi_list={pi_list}, phi_list={phi_list}")
            return pi_list, phi_list

    print(
        f"Converged after {t} iterations with h(theta) ="
        f"{h_theta_t:.6f} and h(theta_t1) = {h_theta_t1:.6f}"
    )
    print(f"Final parameters: pi_list={pi_list}, phi_list={phi_list}")

    return pi_list, phi_list
