import numpy as np
import pytest

from evtpooling import (
    ExtractError,
    h_theta,
    integrand_i,
    non_parametric_density,
    optimize_phi_i,
    optimize_pi_i,
    state_probability,
)

# --------------------
# Fixtures (mock data)
# --------------------


# Mock dependencies
def mock_state_probability(x, i, theta):
    return 0.5  # constant valid value


def mock_non_parametric_density(x, alpha_list, theta):
    return 0.8  # constant valid density


@pytest.fixture
def mock_theta():
    pi_list = [0.2, 0.2, 0.6]
    phi_list = [(0.0, 1.0), (3.0, 1.0), (3.0, 2.0)]

    return (pi_list, phi_list)


@pytest.fixture
def mock_validation_data():
    alpha_list = [0.0, 3.0, 9.0]
    pi_list = [0.2, 0.2, 0.6]
    phi_list = [(0.0, 1.0), (3.0, 1.0), (3.0, 2.0)]

    return alpha_list, (pi_list, phi_list)


@pytest.fixture
def mock_data2():
    alpha_list = [0.1, 0.2, 0.3]
    theta = ([0.5, 0.5], [(1.0, 0.5), (3.0, 1.0)])

    return alpha_list, theta


# Patch actual implementations with mocks
@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    monkeypatch.setattr("evtpooling.pooling.hmix.state_probability", mock_state_probability)
    monkeypatch.setattr(
        "evtpooling.pooling.hmix.non_parametric_density", mock_non_parametric_density
    )


@pytest.fixture
def dummy_theta():
    pi_list = [0.2, 0.8]
    phi_list = [(1.0, 0.5), (3.0, 1.0)]
    return ([1.0, 1.1, 1.2, 1.3, 1.2, 1.3, 3.0, 2.8, 2.7, 3.5, 2.7, 3.0], (pi_list, phi_list))


@pytest.fixture
def known_mixture_data():
    np.random.seed(42)
    # True parameters
    true_pi = [0.4, 0.6]
    true_phi = [(2.0, 0.5), (5.0, 1.0)]

    # Sample data from the known mixture
    n_samples = 100
    n1 = int(n_samples * true_pi[0])
    n2 = n_samples - n1

    samples_1 = np.random.normal(loc=true_phi[0][0], scale=true_phi[0][1], size=n1)
    samples_2 = np.random.normal(loc=true_phi[1][0], scale=true_phi[1][1], size=n2)
    alpha_list = np.concatenate([samples_1, samples_2]).tolist()

    return alpha_list, true_pi, true_phi


# --------------------
# Tests
# --------------------


def test_state_probability(mock_theta):
    probability = state_probability(0.0, 0, mock_theta)

    assert isinstance(probability, float), "The result should be a float."
    assert 0.0 <= probability <= 1.0, "The probability should be between 0 and 1."
    assert probability == pytest.approx(0.667517, rel=1e-5), (
        "The calculated probability for x=0.0 and i=0 does not match the expected value."
    )
    assert state_probability(3.0, 0, mock_theta) == pytest.approx(0.00442394, rel=1e-5), (
        "The calculated probability for x=3.0 and i=0 does not match the expected value."
    )
    assert state_probability(0.0, 2, mock_theta) == pytest.approx(0.325066, rel=1e-5), (
        "The calculated probability for x=0.0 and i=2 does not match the expected value."
    )
    assert state_probability(3.0, 2, mock_theta) == pytest.approx(0.597345, rel=1e-5), (
        "The calculated probability for x=3.0 and i=2 does not match the expected value."
    )


def test_non_parametric_density(mock_validation_data, monkeypatch):
    # Unpatch state_probability just for this test
    monkeypatch.setattr("evtpooling.pooling.hmix.state_probability", state_probability)

    alpha_list, theta = mock_validation_data
    val = non_parametric_density(x=2.0, alpha_list=alpha_list, theta=theta)

    assert theta[0] == [0.2, 0.2, 0.6], "pi_list has changed from expected"
    assert np.isclose(val, 0.0950433545, atol=1e-6), (
        "g_hat should return the expected value for x=2.0"
    )


def test_g_hat_edge_case_empty_input(mock_validation_data):
    alpha_list = []
    _, theta = mock_validation_data
    x = 2.0
    with pytest.raises(ZeroDivisionError):
        non_parametric_density(x, alpha_list, theta)


def test_integrand_i_basic(mock_validation_data, monkeypatch):
    # Unpatch state_probability just for this test
    monkeypatch.setattr("evtpooling.pooling.hmix.state_probability", state_probability)
    monkeypatch.setattr("evtpooling.pooling.hmix.non_parametric_density", non_parametric_density)

    alpha_list, theta = mock_validation_data
    val = integrand_i(x=2.0, i=0, alpha_list=alpha_list, theta=theta)

    assert isinstance(val, float)
    assert np.isclose(val, 0.01833589, atol=1e-6)


# Edge test: sigma too small (density should still be finite)
def test_integrand_i_narrow_variance(mock_validation_data):
    alpha_example, _ = mock_validation_data
    theta = ([1.0], [(2.0, 1e-6)])
    val = integrand_i(x=2.0, i=0, alpha_list=alpha_example, theta=theta)
    assert np.isfinite(val)


# Error handling test
def test_integrand_i_extract_error(monkeypatch, mock_validation_data):
    alpha_example, theta_example = mock_validation_data

    def failing_state_probability(x, i, theta):
        raise ExtractError("Mock failure")

    monkeypatch.setattr("evtpooling.pooling.hmix.state_probability", failing_state_probability)

    with pytest.raises(ExtractError, match="Error in computing integrand"):
        integrand_i(x=2.0, i=0, alpha_list=alpha_example, theta=theta_example)


def test_h_theta_basic(mock_data2):
    alpha_list, theta = mock_data2
    val = h_theta(alpha_list, theta)

    assert isinstance(val, float), "h_theta should return a float"
    assert val > 0, "h_theta should return a positive value"
    assert np.isfinite(val), "h_theta should return a finite value"
    assert np.isclose(val, 1.276592378761983, atol=1e-6)


def test_h_theta_zero_weight_component():
    # Include a zero-weight component to test stability
    alpha_list = [2.0, 3.0, 4.0]
    theta = ([0.0, 1.0], [(2.0, 0.5), (3.0, 0.5)])
    val = h_theta(alpha_list, theta)

    assert isinstance(val, float)
    assert val > 0
    assert np.isclose(val, 0.9989828114392723, atol=1e-6), (
        "h_theta should handle zero-weight components correctly"
    )


def test_h_theta_input_validation_mismatched_components():
    alpha_list = [2.0, 3.0]
    theta = ([0.5, 0.5], [(1.0, 0.5)])  # Only one φ_i

    with pytest.raises(ValueError, match="Mismatch between number of mixture weights"):
        h_theta(alpha_list, theta)


def test_h_theta_input_validation_empty_alpha():
    alpha_list = []
    theta = ([0.5, 0.5], [(1.0, 0.5), (2.0, 0.5)])

    with pytest.raises(ValueError, match="alpha_list must be non-empty"):
        h_theta(alpha_list, theta)


@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
def test_h_theta_nan_propagation(monkeypatch, mock_data2):
    # Force integrand to return nan
    def broken_integrand(*args, **kwargs):
        return np.nan

    monkeypatch.setattr("evtpooling.pooling.hmix.integrand_i", broken_integrand)

    with pytest.raises(RuntimeError, match="Non-finite integral for component"):
        h_theta(*mock_data2)


def test_optimize_phi_i_returns_finite(dummy_theta, monkeypatch):
    # # Unpatch state_probability just for this test
    # monkeypatch.setattr("evtpooling.pooling.hmix.state_probability", state_probability)
    # monkeypatch.setattr(
    #     "evtpooling.pooling.hmix.non_parametric_density", non_parametric_density
    # )

    alpha_list, theta = dummy_theta
    mu_opt, sigma_opt = optimize_phi_i(
        i=1, alpha_list=alpha_list, theta=theta, bounds_arg=(1.0, 10.0)
    )

    assert isinstance(mu_opt, float)
    assert isinstance(sigma_opt, float)
    assert sigma_opt > 0, "Optimized sigma must be positive"
    assert np.isclose(mu_opt, 5.500001696443344, atol=1e-6), (
        "Optimized mu should match expected values"
    )
    assert np.isclose(sigma_opt, 2.2728663212668305, atol=1e-6), (
        "Optimized sigma should match expected values"
    )


def test_optimize_phi_i_improves_h(dummy_theta):
    alpha_list, theta = dummy_theta

    # Original h(θ)
    original_val = h_theta(alpha_list, theta)

    # Optimize component 0
    mu_opt, sigma_opt = optimize_phi_i(i=0, alpha_list=alpha_list, theta=theta)
    new_phi_list = theta[1].copy()
    new_phi_list[0] = (mu_opt, sigma_opt)
    updated_theta = (theta[0], new_phi_list)

    new_val = h_theta(alpha_list, updated_theta)

    assert new_val >= original_val, "Optimized h(θ) should not decrease"


def test_optimize_phi_i_fails_gracefully(monkeypatch, dummy_theta):
    alpha_list, theta = dummy_theta

    def bad_h_total_int(*args, **kwargs):
        return np.nan

    # Force h_theta to return NaN so optimization fails
    monkeypatch.setattr("evtpooling.pooling.hmix.h_total_int", bad_h_total_int)

    with pytest.raises(RuntimeError, match="Optimization failed"):
        optimize_phi_i(i=0, alpha_list=alpha_list, theta=theta)


def test_optimize_pi_i_returns_finite(dummy_theta):
    alpha_list, theta = dummy_theta
    pi_opt = optimize_pi_i(i=0, alpha_list=alpha_list, theta=theta)

    assert isinstance(pi_opt, float)
    assert 0.0 <= pi_opt <= 1.0
    assert np.isclose(pi_opt, 0.1283517956795529, atol=1e-6)


def sort_by_mu(pi_list, phi_list):
    return zip(*sorted(zip(pi_list, phi_list, strict=False), key=lambda x: x[1][0]), strict=False)


# def test_hmix_converges_to_known_parameters(known_mixture_data, monkeypatch):
#     # Unpatch state_probability just for this test
#     monkeypatch.setattr("evtpooling.pooling.hmix.state_probability", state_probability)
#     monkeypatch.setattr(
#         "evtpooling.pooling.hmix.non_parametric_density", non_parametric_density
#     )

#     alpha_list, true_pi, true_phi = known_mixture_data
#     num_clusters = 2

#     pi_est, phi_est = hmix_func(alpha_list, num_clusters, threshold=1e-4, max_iter=50)

#     # Sort estimated and true components by mean for comparison
#     pi_est, phi_est = sort_by_mu(pi_est, phi_est)
#     true_pi, true_phi = sort_by_mu(true_pi, true_phi)

#     pi_est = list(pi_est)
#     phi_est = list(phi_est)

#     for (pi_e, phi_e), (pi_t, phi_t) in zip(
#         zip(pi_est, phi_est, strict=False), zip(true_pi, true_phi, strict=False), strict=False
#     ):
#         mu_e, sigma_e = phi_e
#         mu_t, sigma_t = phi_t

#         assert np.isclose(pi_e, pi_t, atol=0.15), f"Mixing coefficient mismatch: {pi_e} vs {pi_t}"
#         assert np.isclose(mu_e, mu_t, atol=0.5), f"Mean mismatch: {mu_e} vs {mu_t}"
#         assert np.isclose(sigma_e, sigma_t, atol=0.5), f"Std mismatch: {sigma_e} vs {sigma_t}"

#     print("Test passed: Estimated parameters are close to the true mixture components.")
