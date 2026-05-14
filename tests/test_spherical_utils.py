import warnings
from functools import partial
from typing import List, Tuple

import time
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.optim import Adam

from spherics.spherical_utils import (
    spherical_fb8, spherical_logistic, spherical_gaussian, nasg, nasg_gabor,
    spherical_fb6, spherical_cauchy, vMF, spherical_fb4, spherical_beta,
    ltc, asg,
)

def to_uni_dist(f):
    """Wrap f as a single-component (n_gaussians=1) distribution."""
    return partial(f, n_gaussians=1)


def sample_uniform_directions_on_sphere_torch(
    num_samples: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """Sample directions uniformly from the unit sphere surface."""
    phi       = torch.rand(num_samples, device=device) * 2 * torch.pi
    cos_theta = torch.rand(num_samples, device=device) * 2 - 1
    sin_theta = torch.sqrt(1 - cos_theta ** 2)
    x = sin_theta * torch.cos(phi)
    y = sin_theta * torch.sin(phi)
    z = cos_theta
    return torch.stack((x, y, z), dim=1)


def assert_one(errors: List[float]) -> None:
    """Assert that the mean integral estimate is close to 1."""
    errors = np.array(errors)
    assert np.abs(errors.mean() - 1.0) < 0.001, f"Normalization error: {errors.mean():.6f}"
    print("\033[92m.\033[0m", end="", flush=True)


def random_domain(domains: List[Tuple[float, float]], n: int = 1) -> torch.Tensor:
    return torch.stack(
        [torch.rand(n) * (high - low) + low for low, high in domains], dim=0
    ).view(-1)


def integrate(
    f,
    coeff_domains: List[Tuple[float, float]],
    num_tests: int = 1000,
    num_samples: int = 10000,
    n_gaussians: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    plot: bool = True,
) -> List[float]:
    """Test normalization via Monte Carlo integration over the sphere.

    Args:
        f: Distribution ``f(v, coeffs, n_gaussians)``.
        coeff_domains: ``(min, max)`` per parameter.
        num_tests: Random coefficient sets.
        num_samples: Sphere samples per test.
        n_gaussians: Mixture components.
        device: Compute device.

    Returns:
        Integral estimates (approx 1.0 for normalized distributions).
    """
    errors = []
    infs   = 0

    for _ in range(num_tests):
        param_vals = torch.stack(
            [torch.rand(n_gaussians) * (high - low) + low for low, high in coeff_domains],
            dim=1,
        )  # (n_gaussians, n_params_per_lobe)
        coeffs     = param_vals.reshape(-1).to(device)
        coeffs     = coeffs.repeat(num_samples, 1)
        directions = sample_uniform_directions_on_sphere_torch(num_samples, device=device)

        f_values = f(directions, coeffs, n_gaussians=n_gaussians)
        assert f_values.shape == (num_samples,)

        integral = f_values.mean() * 4 * np.pi / n_gaussians
        if torch.isinf(integral) or torch.isnan(integral):
            infs += 1
            continue
        errors.append(integral.item())

    if infs > 0:
        warnings.warn(f"Inf/NaN rate: {infs / num_tests * 100:.2f}%")

    if plot:
        plt.hist(errors, bins=50, range=(-1, 5))
        plt.show()
    return errors


def train_sd(
    spherical_distribution,
    coeff_domains: List[Tuple[float, float]],
    n: int,
) -> None:
    """Optimize a mixture to fit a random target distribution.

    Args:
        spherical_distribution: Distribution function.
        coeff_domains: Per-parameter ``(min, max)`` ranges.
        n: Number of mixture components.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    f      = partial(spherical_distribution, n_gaussians=n)

    target_weights = torch.tensor(
        [[random_domain([domain])[0] for _ in range(n) for domain in coeff_domains]],
        device=device, dtype=torch.float32,
    )
    directions    = sample_uniform_directions_on_sphere_torch(100000, device=device)
    target_values = f(directions, target_weights)

    weights = (target_weights + torch.randn_like(target_weights)).requires_grad_(True)

    optimizer     = Adam([weights], lr=0.075)
    loss_fn       = torch.nn.MSELoss()
    total_forward = total_backward = 0.0
    total_steps   = 1500

    for step in range(total_steps):
        optimizer.zero_grad()

        t0 = time.time()
        predicted = f(directions, weights)
        total_forward += time.time() - t0

        loss = loss_fn(predicted, target_values)

        t0 = time.time()
        loss.backward()
        total_backward += time.time() - t0

        if (torch.isnan(weights.grad).any() or torch.isinf(weights.grad).any() or
                torch.isnan(predicted).any() or torch.isinf(predicted).any()):
            breakpoint()

        optimizer.step()
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.6f}")

    final_loss = loss.item()
    print(
        f"Final Loss: {final_loss:.4e} "
        f"(forward: {total_forward * 1000 / total_steps:.2f}ms, "
        f"backward: {total_backward * 1000 / total_steps:.2f}ms)"
    )
    assert final_loss < 1e-4, "Spherical distribution failed to converge"


# ---------------------------------------------------------------------------
# Normalization tests
# ---------------------------------------------------------------------------

def test_spherical_gaussian() -> None:
    """Test spherical Gaussian normalization."""
    domains = [(-5, 4), (-5, 5), (-5, 5)]
    assert_one(integrate(to_uni_dist(spherical_gaussian), domains))


def test_spherical_gaussian_convergence() -> None:
    """Test spherical Gaussian mixture convergence."""
    domains = [(-5, 4), (-5, 5), (-5, 5)]
    train_sd(spherical_gaussian, domains, 10)


def test_spherical_beta_convergence() -> None:
    """Test spherical Beta distribution convergence."""
    domains = [(-5, 5), (-5, 5), (-5, 5)]
    train_sd(spherical_beta, domains, 10)


def test_nasg() -> None:
    """Test NASG normalization."""
    domains = [(-5, 5), (-5, 5), (-5, 5), (-5, 4), (-5, 4)]
    assert_one(integrate(to_uni_dist(nasg), domains))


def test_nasg_convergence() -> None:
    """Test NASG mixture convergence."""
    domains = [(-5, 5), (-5, 5), (-5, 5), (-5, 4), (-5, 4)]
    train_sd(nasg, domains, 10)


def test_vMF() -> None:
    """Test von Mises-Fisher normalization."""
    domains = [(-5, 5), (-5, 5), (-5, 4)]
    assert_one(integrate(to_uni_dist(vMF), domains))


def test_vMF_convergence() -> None:
    """Test vMF mixture convergence."""
    domains = [(-5, 5), (-5, 5), (-5, 4)]
    train_sd(vMF, domains, 10)


def test_spherical_logistic() -> None:
    """Test spherical logistic normalization."""
    domains = [(-5, 5), (-5, 5), (-5, 4), (-5, 4)]
    assert_one(integrate(to_uni_dist(spherical_logistic), domains))


def test_spherical_logistic_convergence() -> None:
    """Test spherical logistic mixture convergence."""
    domains = [(-5, 5), (-5, 5), (-5, 4), (-5, 4)]
    train_sd(spherical_logistic, domains, 10)


def test_fb4() -> None:
    """Test FB4 normalization."""
    domains = [(-5, 5), (-5, 5), (-5, 3), (-5, 2)]
    assert_one(integrate(to_uni_dist(spherical_fb4), domains))


def test_fb4_convergence() -> None:
    """Test FB4 mixture convergence."""
    domains = [(-5, 5), (-5, 5), (-5, 2), (-5, 2)]
    train_sd(spherical_fb4, domains, 10)


def test_fb6() -> None:
    """Test FB6 normalization."""
    domains = [(-5, 5), (-5, 5), (-5, 5), (-5, 2), (-5, 2), (-5, 4)]
    assert_one(integrate(to_uni_dist(spherical_fb6), domains))


def test_fb6_convergence() -> None:
    """Test FB6 mixture convergence."""
    domains = [(-5, 5), (-5, 5), (-5, 5), (-5, 3), (-5, 3), (-5, 5)]
    train_sd(spherical_fb6, domains, 10)


def test_fb8() -> None:
    """Test FB8 normalization."""
    domains = [(-5, 5), (-5, 5), (-5, 5), (-5, 3), (-5, 3), (-5, 5), (-5, 5), (-5, 5)]
    assert_one(integrate(to_uni_dist(spherical_fb8), domains))


def test_spherical_cauchy() -> None:
    """Test spherical Cauchy normalization."""
    domains = [(-5, 5), (-5, 5), (-5, 3)]
    assert_one(integrate(to_uni_dist(spherical_cauchy), domains))


def test_nasg_gabor() -> None:
    """Test NASG-Gabor normalization."""
    domains = [(-5, 5), (-5, 5), (-5, 5), (-5, 4), (-5, 4), (-5, 3)]
    assert_one(integrate(to_uni_dist(nasg_gabor), domains))


def test_nasg_gabor_convergence() -> None:
    """Test NASG-Gabor mixture convergence."""
    domains = [(-5, 5), (-5, 5), (-5, 5), (-5, 4), (-5, 4), (-5, 3)]
    train_sd(nasg_gabor, domains, 10)


def test_ltc() -> None:
    """Test LTC normalization."""
    domains = [(-5, 5), (-5, 5), (-5, 5), (-5, 5)]
    assert_one(integrate(to_uni_dist(ltc), domains))


def test_ltc_convergence() -> None:
    """Test LTC mixture convergence."""
    domains = [(-5, 5), (-5, 5), (-5, 5), (-5, 5)]
    train_sd(ltc, domains, 10)


def test_asg_convergence() -> None:
    """Test ASG mixture convergence (no closed-form normalization)."""
    domains = [(-5, 5), (-5, 5), (-5, 5), (-5, 4), (-5, 4)]
    train_sd(asg, domains, 10)


if __name__ == "__main__":
    # test_fb4()
    # test_fb4_convergence()

    test_fb6()
    test_fb6_convergence()

    # test_fb8()
    # test_fb8_convergence()

    # test_spherical_cauchy()
    # test_spherical_cauchy_convergence()

    # test_spherical_logistic()
    # test_spherical_logistic_convergence()

    # test_vMF()
    # test_vMF_convergence()

    # test_spherical_gaussian()
    # test_spherical_gaussian_convergence()

    # test_nasg()
    # test_nasg_convergence()

    # test_spherical_beta()
    # test_spherical_beta_convergence()

    # test_nasg_gabor()
    # test_nasg_gabor_convergence()

    # test_ltc()
    # test_ltc_convergence()

    # test_asg_convergence()
