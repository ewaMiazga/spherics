from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch.special import erf
from typing import *


def get_param(index: int, params: torch.Tensor, n_gaussians: int, l: int = 1) -> torch.Tensor:
    """Extract parameter block for a given index from the flat coefficient tensor.

    Assumes a lobe-first (interleaved) memory layout:
        [lobe₀_p₀ lobe₀_p₁ … lobe₀_pK | lobe₁_p₀ lobe₁_p₁ … | … | lobeN_p₀ …]

    Args:
        index: Parameter slot index (0-based within each lobe).
        params: Flat coefficient tensor of shape
            ``(B, n_params_per_lobe * n_gaussians)``.
        n_gaussians: Number of mixture components.
        l: Number of consecutive values per component to extract (e.g. 3 for RGB).

    Returns:
        Tensor of shape ``(B, n_gaussians * l)``.
    """
    n_params_per_lobe = params.shape[1] // n_gaussians
    if l == 1:
        return params[:, index::n_params_per_lobe]
    lobe_starts = torch.arange(n_gaussians, device=params.device) * n_params_per_lobe + index
    all_indices = (lobe_starts.unsqueeze(1) + torch.arange(l, device=params.device)).reshape(-1)
    return params[:, all_indices]


def dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sum(a * b, dim=2, keepdim=True)


def get_basis_parameterized(
    cosθ: torch.Tensor, cosϕ: torch.Tensor, cosτ: torch.Tensor
) -> list:
    clamp_min = -0.999999
    clamp_max = 0.999999
    cosθ = torch.clamp(cosθ, clamp_min, clamp_max)
    cosϕ = torch.clamp(cosϕ, clamp_min, clamp_max)
    cosτ = torch.clamp(cosτ, clamp_min, clamp_max)

    sinθ = torch.sqrt(1.0 - cosθ * cosθ)
    sinϕ = torch.sqrt(1.0 - cosϕ * cosϕ)
    sinτ = torch.sqrt(1.0 - cosτ * cosτ)

    x = torch.stack(
        [cosθ * cosϕ * cosτ - sinθ * sinτ,
         sinθ * cosϕ * cosτ + cosθ * sinτ,
         -sinϕ * cosτ],
        dim=2,
    )
    z = torch.stack(
        [cosθ * sinϕ,
         sinθ * sinϕ,
         cosϕ],
        dim=2,
    )
    return [x, z]


def sinch(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable sinch(x) = sinh(x)/x, with sinch(0) = 1."""
    x_clamped = torch.clamp(x, min=-40.0, max=40.0)
    safe_x = torch.where(torch.abs(x_clamped) < 1e-4, torch.ones_like(x_clamped), x_clamped)
    result = torch.sinh(x_clamped) / safe_x
    max_sinch = torch.sinh(torch.tensor(40.0, device=x.device, dtype=x.dtype)) / 40.0
    result = torch.where(torch.abs(x) >= 40.0, max_sinch * torch.ones_like(x), result)
    return torch.where(torch.abs(x_clamped) < 1e-4, torch.ones_like(x_clamped), result)


# ---------------------------------------------------------------------------
# NASG – Non-linearly Anisotropic Spherical Gaussian
# ---------------------------------------------------------------------------

def nasg(
    v: torch.Tensor,
    coeffs: torch.Tensor,
    n_gaussians: int,
    normalized: bool = True,
) -> torch.Tensor:
    """Non-linearly Anisotropic Spherical Gaussian (NASG).

    Parameter layout (per component):
        0: cosθ, 1: cosϕ, 2: cosτ  – frame orientation
        3: λ (log-space), 4: a (log-space)

    Args:
        v: Unit direction vectors, shape ``(B, 3)``.
        coeffs: Flat coefficient tensor, shape ``(B, 5 * n_gaussians)``.
        n_gaussians: Number of mixture components.
        normalized: If ``True`` (default) applies the analytic normalization
            constant so that each component integrates to 1 over the sphere.

    Returns:
        PDF tensor of shape ``(B,)``.
    """
    param = partial(get_param, params=coeffs, n_gaussians=n_gaussians)
    v = v.unsqueeze(1)

    eps = 5e-6
    x, z = get_basis_parameterized(param(0), param(1), param(2))

    λ = torch.clamp(torch.exp(param(3).unsqueeze(2)), max=1e4)
    a = torch.clamp(torch.exp(param(4).unsqueeze(2)), max=1e4)

    vz = dot(v, z)
    mask_one  = vz >= 1.0 - 1e-7
    mask_zero = vz <= -1.0 + 1e-7
    valid = ~mask_one & ~mask_zero

    placeholder = torch.zeros_like(vz)

    K_base = (vz[valid] + 1.0) * 0.5
    K_exp  = eps + (a * (dot(v, x) ** 2.0))[valid] / (1.0 - vz[valid] ** 2.0)
    exp    = torch.pow(K_base, K_exp)

    norm = inv_nasg_norm(λ, a) if normalized else torch.ones_like(λ)
    pdf  = torch.exp(2.0 * (λ * valid)[valid] * (exp * K_base - 1.0)) * exp * (norm * valid)[valid]

    placeholder[valid] = pdf
    placeholder = torch.where(~mask_one,  placeholder, 1.0)
    pdf         = torch.where(~mask_zero, placeholder, 0.0)

    return pdf.squeeze(-1).sum(dim=1)


def inv_nasg_norm(λ: torch.Tensor, a: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Inverse of the NASG normalization constant (i.e. 1 / Z)."""
    num   = 2.0 * torch.pi * (1.0 + eps - torch.exp(-2.0 * λ))
    denom = λ * torch.sqrt(1.0 + a)
    return denom / num


# ---------------------------------------------------------------------------
# NASG-Gabor
# ---------------------------------------------------------------------------

def nasg_gabor(
    v: torch.Tensor,
    coeffs: torch.Tensor,
    n_gaussians: int,
    normalized: bool = True,
) -> torch.Tensor:
    """NASG modulated with a cosine (Gabor-like) term.

    Parameter layout (per component):
        0: cosθ, 1: cosϕ, 2: cosτ  – frame orientation
        3: λ (log-space), 4: a (log-space)
        5: k (frequency, tanh-activated)

    Args:
        v: Unit direction vectors, shape ``(B, 3)``.
        coeffs: Flat coefficient tensor, shape ``(B, 6 * n_gaussians)``.
        n_gaussians: Number of mixture components.
        normalized: If ``True`` applies the NASG normalization constant.

    Returns:
        PDF tensor of shape ``(B,)``.
    """
    param = partial(get_param, params=coeffs, n_gaussians=n_gaussians)
    v = v.unsqueeze(1)

    eps = 5e-6
    x, z = get_basis_parameterized(param(0), param(1), param(2))

    λ = torch.clamp(torch.exp(param(3).unsqueeze(2)), max=1e4)
    a = torch.clamp(torch.exp(param(4).unsqueeze(2)), max=1e4)
    k = (torch.tanh(param(5).unsqueeze(2)) + 1.0) * 20.0

    vz = dot(v, z)
    mask_one  = vz >= 1.0 - 1e-7
    mask_zero = vz <= -1.0 + 1e-7
    valid = ~mask_one & ~mask_zero

    placeholder = torch.zeros_like(vz)

    K_base = (vz[valid] + 1.0) * 0.5
    K_exp  = eps + (a * (dot(v, x) ** 2.0))[valid] / (1.0 - vz[valid] ** 2.0)
    exp    = torch.pow(K_base, K_exp)

    cosine_term = torch.cos(k * dot(v, x))[valid]
    norm = inv_nasg_norm(λ, a) if normalized else torch.ones_like(λ)

    pdf = (
        torch.exp(2.0 * (λ * valid)[valid] * (exp * K_base - 1.0))
        * exp
        * ((1.0 + cosine_term) * 0.5)
        * (norm * valid)[valid]
    )

    placeholder[valid] = pdf
    placeholder = torch.where(~mask_one,  placeholder, 1.0)
    pdf         = torch.where(~mask_zero, placeholder, 0.0)

    return pdf.squeeze(-1).sum(dim=1)


def nasg_gabor_normalization(
    coeffs: torch.Tensor,
    n_gaussians: int,
) -> torch.Tensor:
    """Compute the Gabor normalization term for NASG-Gabor.

    Args:
        coeffs: Flat coefficient tensor.
        n_gaussians: Number of mixture components.

    Returns:
        Normalization tensor of shape ``(B, n_gaussians)``.
    """
    param = partial(get_param, params=coeffs, n_gaussians=n_gaussians)

    λ = torch.clamp(torch.exp(param(3).unsqueeze(2)), max=1e4)
    a = torch.clamp(torch.exp(param(4).unsqueeze(2)), max=1e4)
    k = (torch.tanh(param(5).unsqueeze(2)) + 1.0) * 20.0

    Lambda    = λ * (1.0 + a)
    sqrt_arg  = torch.clamp(Lambda ** 2 - torch.abs(k) ** 2, min=1e-8)
    sinch_val = sinch(torch.sqrt(sqrt_arg))

    Lambda_safe     = torch.clamp(Lambda, min=1e-8, max=1e6)
    sinch_safe      = torch.where(torch.isfinite(sinch_val), sinch_val, torch.ones_like(sinch_val))
    exp_neg_Lambda  = torch.exp(-Lambda_safe)
    exp_neg2_Lambda = torch.exp(-2.0 * Lambda_safe)

    norm_num = torch.clamp(2.0 * Lambda_safe * exp_neg_Lambda * sinch_safe, min=-1e6, max=1e6)
    norm_num = torch.where(torch.isfinite(norm_num), norm_num, torch.zeros_like(norm_num))
    norm_den = torch.where(torch.isfinite(1.0 - exp_neg2_Lambda),
                           1.0 - exp_neg2_Lambda, torch.ones_like(exp_neg2_Lambda))
    norm_den = torch.where(torch.abs(norm_den) < 1e-8,
                           torch.full_like(norm_den, 1e-8), norm_den)
    norm_raw = 1.0 + norm_num / norm_den
    norm_term = torch.where(
        torch.isfinite(norm_raw) & (norm_raw > 0) & (norm_raw < 1e6),
        norm_raw,
        torch.ones_like(norm_raw),
    )
    return norm_term[..., 0]


# ---------------------------------------------------------------------------
# LTC – Linearly Transformed Cosines
# ---------------------------------------------------------------------------

def ltc(
    v: torch.Tensor,
    coeffs: torch.Tensor,
    n_gaussians: int,
    normalized: bool = True,
) -> torch.Tensor:
    """Linearly Transformed Cosine distribution.

    Parameter layout (per component):
        0: a, 1: b, 2: c, 3: d  – matrix entries (tanh-activated)

    Args:
        v: Unit direction vectors, shape ``(B, 3)``.
        coeffs: Flat coefficient tensor, shape ``(B, 4 * n_gaussians)``.
        n_gaussians: Number of mixture components.
        normalized: If ``True`` applies the Jacobian-based normalization;
            if ``False`` returns the raw cosine term without the Jacobian.

    Returns:
        PDF tensor of shape ``(B,)``.
    """
    param = partial(get_param, params=coeffs, n_gaussians=n_gaussians)
    v = v.unsqueeze(1)

    eps = 1e-6
    a = torch.tanh(param(0).unsqueeze(2))
    b = torch.tanh(param(1).unsqueeze(2))
    c = torch.tanh(param(2).unsqueeze(2))
    d = torch.tanh(param(3).unsqueeze(2))

    zeros = torch.zeros_like(a)
    ones  = torch.ones_like(a)

    M = torch.stack(
        [torch.stack([a,     zeros, b],     dim=-1),
         torch.stack([zeros, c,     zeros], dim=-1),
         torch.stack([d,     zeros, ones],  dim=-1)],
        dim=-2,
    ).squeeze(2)

    M    = M + 1e-6 * torch.eye(3, device=M.device)
    Minv = torch.linalg.pinv(M)

    w      = v.unsqueeze(-1)
    Minv_w = torch.matmul(Minv, w).squeeze(-1)
    norm_v = torch.norm(Minv_w, dim=-1, keepdim=True) + eps
    w_     = Minv_w / norm_v

    cos_term = torch.clamp(w_[..., 2], min=0.0) / torch.pi

    if normalized:
        det      = torch.abs(torch.det(Minv)).unsqueeze(-1)
        jacobian = det / (norm_v ** 3)
        D        = cos_term * jacobian.squeeze(-1)
    else:
        D = cos_term

    return D.sum(dim=1)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def gauss_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def log_factorial(x: torch.Tensor) -> torch.Tensor:
    return torch.lgamma(x + 1)


def factorial(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(log_factorial(x))


def gamma(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(torch.lgamma(x))


def pochhammer(a: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return gamma(a + n) / gamma(a)


def euler2rotmat(cx: torch.Tensor, cy: torch.Tensor, cz: torch.Tensor) -> tuple:
    clamp_min = -0.999999
    clamp_max = 0.999999
    cx = torch.clamp(cx, clamp_min, clamp_max)
    cy = torch.clamp(cy, clamp_min, clamp_max)
    cz = torch.clamp(cz, clamp_min, clamp_max)

    sx = torch.sqrt(1.0 - cx * cx)
    sy = torch.sqrt(1.0 - cy * cy)
    sz = torch.sqrt(1.0 - cz * cz)

    x = torch.stack([cy * cz, cx * sz + sx * sy * cz, sx * sz - cx * sy * cz], dim=2)
    y = torch.stack([-cy * sz, cx * cz - sx * sy * sz, sx * cz + cx * sy * sz], dim=2)
    z = torch.stack([sy, -sx * cy, cx * cy], dim=2)
    return x, y, z


def polar2cart(cx: torch.Tensor, cy: torch.Tensor) -> torch.Tensor:
    clamp_min = -0.999999
    clamp_max = 0.999999
    cx = torch.clamp(cx, clamp_min, clamp_max)
    cy = torch.clamp(cy, clamp_min, clamp_max)
    sx = torch.sqrt(1.0 - cx * cx)
    sy = torch.sqrt(1.0 - cy * cy)
    return torch.stack([sx * cy, sx * sy, cx], dim=2)


def cart2polar(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    r     = torch.sqrt(x * x + y * y + z * z)
    theta = torch.acos(z / r)
    phi   = torch.atan2(y, x)
    return torch.stack([theta, phi], dim=0)


def get_dirs(env_w: int = 800, env_h: int = 400) -> torch.Tensor:
    Az = ((torch.arange(env_w, dtype=torch.float32) + 0.5) / env_w - 0.5) * 2 * torch.pi
    El = ((torch.arange(env_h, dtype=torch.float32) + 0.5) / env_h) * torch.pi / 2.0
    Az, El = torch.meshgrid(Az, El)
    lx = torch.sin(El) * torch.cos(Az)
    ly = torch.sin(El) * torch.sin(Az)
    lz = torch.cos(El)
    return torch.stack((lx, ly, lz), dim=-1).permute(1, 0, 2).cuda()


# ---------------------------------------------------------------------------
# Spherical distribution functions
# ---------------------------------------------------------------------------

def vMF(
    v: torch.Tensor,
    coeffs: torch.Tensor,
    n_gaussians: int,
    normalized: bool = True,
) -> torch.Tensor:
    """von Mises–Fisher distribution on the sphere.

    Reference: https://www.jstor.org/stable/3213566

    Parameter layout (per component):
        0: cos(polar angle) of μ
        1: cos(azimuthal angle) of μ
        2: concentration κ (log-space)

    Args:
        v: Unit direction vectors, shape ``(B, 3)``.
        coeffs: Flat coefficient tensor, shape ``(B, 3 * n_gaussians)``.
        n_gaussians: Number of mixture components.
        normalized: If ``True`` applies the vMF normalization constant
            ``κ / (4π sinh(κ))``.

    Returns:
        PDF tensor of shape ``(B,)``.
    """
    param = partial(get_param, params=coeffs, n_gaussians=n_gaussians)
    v = v.unsqueeze(1)

    μ = polar2cart(param(0), param(1))
    κ = torch.exp(param(2)).unsqueeze(2) + 1e-6

    exp_term = torch.exp(κ * dot(μ, v))
    if normalized:
        pdf = (κ / (4.0 * torch.pi * torch.sinh(κ))) * exp_term
    else:
        pdf = exp_term

    return pdf.squeeze(-1).sum(dim=1)


def spherical_beta(
    v: torch.Tensor,
    coeffs: torch.Tensor,
    n_gaussians: int,
    normalized: bool = True,
) -> torch.Tensor:
    """Spherical Beta distribution.

    Reference: https://arxiv.org/pdf/2501.18630

    Parameter layout (per component):
        0: cos(polar angle) of μ
        1: cos(azimuthal angle) of μ
        2: shape parameter β

    Args:
        v: Unit direction vectors, shape ``(B, 3)``.
        coeffs: Flat coefficient tensor, shape ``(B, 3 * n_gaussians)``.
        n_gaussians: Number of mixture components.
        normalized: If ``True`` divides by ``2π / (exponent + 1)``.

    Returns:
        PDF tensor of shape ``(B,)``.
    """
    param = partial(get_param, params=coeffs, n_gaussians=n_gaussians)
    v = v.unsqueeze(1)

    μ        = polar2cart(param(0), param(1))
    β        = param(2).unsqueeze(2)
    exponent = 4.0 * torch.exp(β)
    pdf      = torch.pow(torch.clamp(dot(μ, v), 1e-6, 1.0), exponent)

    if normalized:
        pdf = pdf / (2.0 * torch.pi / (exponent + 1))

    return pdf.squeeze(-1).sum(dim=1)


def spherical_logistic(
    v: torch.Tensor,
    coeffs: torch.Tensor,
    n_gaussians: int,
    normalized: bool = True,
) -> torch.Tensor:
    """Spherical Logistic distribution.

    Reference: https://link.springer.com/article/10.1007/s40304-018-00171-2

    Parameter layout (per component):
        0: cos(polar angle) of μ
        1: cos(azimuthal angle) of μ
        2: concentration k (softplus-activated)
        3: tail parameter b (softplus-activated, shifted by 1)

    Args:
        v: Unit direction vectors, shape ``(B, 3)``.
        coeffs: Flat coefficient tensor, shape ``(B, 4 * n_gaussians)``.
        n_gaussians: Number of mixture components.
        normalized: If ``True`` includes the analytic normalization constant C.

    Returns:
        PDF tensor of shape ``(B,)``.
    """
    param = partial(get_param, params=coeffs, n_gaussians=n_gaussians)
    v = v.unsqueeze(1)

    μ = polar2cart(param(0), param(1))
    k = torch.clamp(F.softplus(param(2)).unsqueeze(2), min=1e-6, max=1e6)
    b = torch.clamp(1.0 + F.softplus(param(3)).unsqueeze(2), min=1e-6, max=1e6)

    exp_term = torch.exp(k * dot(μ, v))
    kernel   = exp_term / ((b - 1.0 + exp_term) ** 2.0)

    if normalized:
        num = k * (b ** 2 + 2.0 * (b - 1.0) * (torch.cosh(k) - 1.0))
        C   = num / (4 * torch.pi * torch.sinh(k))
        pdf = C * kernel
    else:
        pdf = kernel

    return pdf.squeeze(-1).sum(dim=1)


def spherical_gaussian(
    v: torch.Tensor,
    coeffs: torch.Tensor,
    n_gaussians: int,
    normalized: bool = True,
) -> torch.Tensor:
    """Spherical Gaussian distribution.

    Parameter layout (per component):
        0: cos(polar angle) of μ
        1: cos(azimuthal angle) of μ
        2: bandwidth λ (log-space)

    Args:
        v: Unit direction vectors, shape ``(B, 3)``.
        coeffs: Flat coefficient tensor, shape ``(B, 3 * n_gaussians)``.
        n_gaussians: Number of mixture components.
        normalized: If ``True`` divides by ``(2π/λ)(1 − exp(−2λ))``.

    Returns:
        PDF tensor of shape ``(B,)``.
    """
    param = partial(get_param, params=coeffs, n_gaussians=n_gaussians)
    v = v.unsqueeze(1)

    μ      = polar2cart(param(0), param(1))
    λ      = torch.clamp(torch.exp(param(2)).unsqueeze(2), min=1e-6, max=1e4)
    kernel = torch.exp(λ * (dot(μ, v) - 1.0))

    if normalized:
        norm = (2.0 * torch.pi / λ) * (1.0 - torch.exp(-2.0 * λ))
        pdf  = kernel / norm
    else:
        pdf = kernel

    return pdf.squeeze(-1).sum(dim=1)


def spherical_fb6(
    v: torch.Tensor,
    coeffs: torch.Tensor,
    n_gaussians: int,
    normalized: bool = True,
) -> torch.Tensor:
    """Fisher–Bingham FB6 distribution.

    Parameter layout (per component):
        0–2: Euler angles of the rotation matrix (cosine-parameterized)
        3: concentration k (log-space)
        4: anisotropy β (log-space)
        5: asymmetry η (tanh-activated)

    Args:
        v: Unit direction vectors, shape ``(B, 3)``.
        coeffs: Flat coefficient tensor, shape ``(B, 6 * n_gaussians)``.
        n_gaussians: Number of mixture components.
        normalized: If ``True`` divides by the partition function c.

    Returns:
        PDF tensor of shape ``(B,)``.
    """
    param = partial(get_param, params=coeffs, n_gaussians=n_gaussians)
    v = v.unsqueeze(1)

    rotx, roty, rotz = euler2rotmat(param(0), param(1), param(2))
    k   = torch.clamp(torch.exp(param(3)).unsqueeze(2), min=1e-6, max=50)
    β   = torch.clamp(torch.exp(param(4)).unsqueeze(2), min=1e-6, max=50)
    eta = torch.tanh(param(5)).unsqueeze(2)

    kernel = torch.exp(k * dot(rotz, v) + β * (dot(rotx, v) ** 2.0 - eta * (dot(roty, v) ** 2.0)))

    if normalized:
        switch = k <= 2.0 * β
        c      = torch.zeros_like(k)
        hyp    = hyp1f1(
            0.5, 1.0,
            β[switch] * (1.0 + eta[switch]) * (k[switch] ** 2 / (4 * β[switch] ** 2) - 1),
            20,
        )
        c[switch]  = (
            2 * torch.pi
            * torch.exp(β[switch] * (1 + k[switch] ** 2.0 / (4 * β[switch] ** 2.0)))
            * torch.sqrt(torch.pi / β[switch])
            * hyp
        )
        c[~switch] = (
            2 * torch.pi
            * torch.exp(k[~switch])
            / torch.sqrt((k[~switch] - 2 * β[~switch]) * (k[~switch] + 2 * β[~switch] * eta[~switch]))
        )
        pdf = kernel / c
    else:
        pdf = kernel

    return pdf.squeeze(-1).sum(dim=1)


def spherical_cauchy(
    v: torch.Tensor,
    coeffs: torch.Tensor,
    n_gaussians: int,
    normalized: bool = True,
) -> torch.Tensor:
    """Spherical Cauchy distribution.

    Parameter layout (per component):
        0: cos(polar angle) of μ
        1: cos(azimuthal angle) of μ
        2: concentration ρ (sigmoid-activated to (0, 1))

    Args:
        v: Unit direction vectors, shape ``(B, 3)``.
        coeffs: Flat coefficient tensor, shape ``(B, 3 * n_gaussians)``.
        n_gaussians: Number of mixture components.
        normalized: If ``True`` divides by 4π.

    Returns:
        PDF tensor of shape ``(B,)``.
    """
    param = partial(get_param, params=coeffs, n_gaussians=n_gaussians)
    v = v.unsqueeze(1)

    μ       = polar2cart(param(0), param(1))
    rho     = torch.sigmoid(param(2).unsqueeze(2))
    eps     = 1e-6
    kernel  = ((1.0 - rho ** 2) / (1.0 + rho ** 2 - 2.0 * rho * dot(μ, v) + eps)) ** 2.0

    pdf = kernel / (4.0 * torch.pi) if normalized else kernel

    return pdf.squeeze(-1).sum(dim=1)


def spherical_fb4(
    v: torch.Tensor,
    coeffs: torch.Tensor,
    n_gaussians: int,
    normalized: bool = True,
) -> torch.Tensor:
    """Fisher–Bingham FB4 distribution.

    Reference: https://www.jstor.org/stable/pdf/2335218.pdf

    Parameter layout (per component):
        0: cos(polar angle) of μ
        1: cos(azimuthal angle) of μ
        2: scale parameter (softplus-activated)
        3: offset parameter β (softplus-activated)

    Args:
        v: Unit direction vectors, shape ``(B, 3)``.
        coeffs: Flat coefficient tensor, shape ``(B, 4 * n_gaussians)``.
        n_gaussians: Number of mixture components.
        normalized: Accepted for API consistency; closed-form normalization
            is not yet implemented.

    Returns:
        PDF tensor of shape ``(B,)``.
    """
    param = partial(get_param, params=coeffs, n_gaussians=n_gaussians)
    v = v.unsqueeze(1)

    μ   = polar2cart(param(0), param(1))
    β   = F.softplus(param(3)).unsqueeze(2)
    tau = β * 2.0 + F.softplus(param(2)).unsqueeze(2)
    pdf = torch.exp(-tau * (dot(μ, v) - β) ** 2)

    return pdf.squeeze(-1).sum(dim=1)


def spherical_fb8(
    v: torch.Tensor,
    coeffs: torch.Tensor,
    n_gaussians: int,
    normalized: bool = True,
    eps: float = 1e-12,
    s: int = 3,
    max_iter: int = 2,
) -> torch.Tensor:
    """Fisher–Bingham FB8 distribution.

    Parameter layout (per component):
        0–2: Euler angles of the rotation matrix
        3: k (softplus-activated)
        4: beta (softplus-activated)
        5: eta (tanh-activated)
        6: cos(polar angle) of ν
        7: cos(azimuthal angle) of ν

    Args:
        v: Unit direction vectors, shape ``(B, 3)``.
        coeffs: Flat coefficient tensor, shape ``(B, 8 * n_gaussians)``.
        n_gaussians: Number of mixture components.
        normalized: If ``True`` computes the partition function c8 and divides
            by it; if ``False`` returns the unnormalized kernel.

    Returns:
        PDF tensor of shape ``(B,)``.
    """
    param  = partial(get_param, params=coeffs, n_gaussians=n_gaussians)
    device = v.device

    rotx, roty, rotz = euler2rotmat(param(0), param(1), param(2))
    k    = torch.clamp(F.softplus(param(3)).unsqueeze(-1), min=1e-6, max=50)
    beta = torch.clamp(F.softplus(param(4)).unsqueeze(-1), min=1e-6, max=50)
    eta  = torch.tanh(param(5)).unsqueeze(-1)
    mu   = polar2cart(param(6), param(7))

    v_exp  = v.unsqueeze(1)
    mat    = torch.stack([rotx, roty, rotz], dim=-1)
    mu_rot = torch.einsum("ngij,ngj->ngi", mat, mu)

    term1  = dot(k * (mu_rot - rotx), v_exp)
    term2  = beta * (dot(roty, v_exp) ** 2 - eta * dot(rotz, v_exp) ** 2)
    kernel = torch.exp(term1 + term2)

    if normalized:
        c8  = compute_c8_hybrid(kappa=k[0], beta=beta[0], eta=eta[0], nu=mu[0],
                                 L_max=2, K_max=2, J_max=3, block_size=2).unsqueeze(-1)
        pdf = kernel / c8
    else:
        pdf = kernel

    return pdf.squeeze(-1).sum(dim=1)


def asg(
    v: torch.Tensor,
    coeffs: torch.Tensor,
    n_gaussians: int,
    normalized: bool = True,
) -> torch.Tensor:
    """Anisotropic Spherical Gaussian (ASG).

    ``G(v; [x,y,z], [λ,μ]) = max(v·z, 0) · exp(−λ(v·x)² − μ(v·y)²)``

    Note: No closed-form normalization constant exists; the ``normalized``
    parameter is accepted for API consistency but has no effect.

    Parameter layout (per component):
        0: cosθ, 1: cosϕ, 2: cosτ  – frame orientation
        3: λ (softplus-activated), 4: μ (softplus-activated)

    Args:
        v: Unit direction vectors, shape ``(B, 3)``.
        coeffs: Flat coefficient tensor, shape ``(B, 5 * n_gaussians)``.
        n_gaussians: Number of mixture components.
        normalized: Accepted for API consistency; currently unused.

    Returns:
        PDF tensor of shape ``(B,)``.
    """
    param = partial(get_param, params=coeffs, n_gaussians=n_gaussians)
    v = v.unsqueeze(1)

    x, z = get_basis_parameterized(param(0), param(1), param(2))
    y = torch.cross(z, x, dim=2)

    λ = F.softplus(param(3).unsqueeze(2))
    μ = F.softplus(param(4).unsqueeze(2))

    pdf = torch.clamp(dot(v, z), min=0.0) * torch.exp(-λ * dot(v, x) ** 2 - μ * dot(v, y) ** 2)

    return pdf.squeeze(-1).sum(dim=1)


def inv_spherical_logistic_norm(k: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Inverse normalization constant for the spherical logistic distribution."""
    numer = 4 * torch.pi * torch.sinh(k)
    denom = k * (b ** 2 + 2.0 * (b - 1.0) * (torch.cosh(k) - 1.0))
    return denom / numer


# ---------------------------------------------------------------------------
# Hypergeometric helper functions
# ---------------------------------------------------------------------------

def hyp1f1(a: float, b: float, x: torch.Tensor, n: int = 40) -> torch.Tensor:
    """Kummer's confluent hypergeometric function ₁F₁(a; b; x).

    Args:
        a: First numerator parameter.
        b: Denominator parameter.
        x: Argument tensor.
        n: Number of series terms.

    Returns:
        Tensor with the same shape as ``x``.
    """
    if len(x) == 0:
        return x
    result = torch.ones_like(x)
    z      = torch.ones_like(x)
    for k in range(n):
        z      *= (a + k) / (b + k) * x / (k + 1)
        result += z
    return result


def hyp0f1_torch(a: float, z: torch.Tensor, max_iter: int = 5, tol: float = 1e-10) -> torch.Tensor:
    """Confluent hypergeometric limit function ₀F₁(; a; z)."""
    term      = torch.ones_like(z)
    result    = term.clone()
    poch      = torch.ones_like(z)
    factorial = 1.0
    for n in range(1, max_iter):
        factorial *= n
        poch      *= a + n - 1
        term       = term * z / (poch * factorial)
        result    += term
        if torch.max(torch.abs(term)) < tol:
            break
    return result


def hyp2f1_torch(
    a: float, b: float, c: float, z: torch.Tensor,
    max_iter: int = 5, tol: float = 1e-10
) -> torch.Tensor:
    """Gauss hypergeometric function ₂F₁(a, b; c; z) via series expansion (|z| < 1).

    Args:
        a, b: Numerator parameters.
        c: Denominator parameter.
        z: Argument tensor.
        max_iter: Maximum number of series terms.
        tol: Convergence tolerance.

    Returns:
        Approximation of ₂F₁ with the same shape as ``z``.
    """
    term   = torch.ones_like(z)
    result = term.clone()
    for n in range(1, max_iter):
        term   *= (a + n - 1) * (b + n - 1) / ((c + n - 1) * n) * z
        result += term
        if torch.abs(term).max() < tol:
            break
    return result


def compute_c8_hybrid(
    kappa: torch.Tensor,
    beta: torch.Tensor,
    eta: torch.Tensor,
    nu: torch.Tensor,
    L_max: int = 2,
    K_max: int = 2,
    J_max: int = 3,
    block_size: int = 2,
) -> torch.Tensor:
    """Approximate the FB8 partition function c8 via block summation.

    Args:
        kappa, beta, eta: Shape ``(G, 1)``.
        nu: Shape ``(G, 3)``.
        L_max, K_max, J_max: Block loop upper bounds.
        block_size: Inner loop range per block.

    Returns:
        c8 tensor of shape ``(G, 1)``.
    """
    device = nu.device
    dtype  = nu.dtype

    nu1 = nu[..., 0:1].clamp(min=1e-6)
    nu2 = nu[..., 1:2].abs().clamp(min=1e-6)
    nu3 = nu[..., 2:3].abs().clamp(min=1e-6)

    c8: torch.Tensor = torch.zeros_like(nu1)

    lgamma_cache: dict = {}

    def lgamma_cached(x: float) -> torch.Tensor:
        if x not in lgamma_cache:
            lgamma_cache[x] = torch.lgamma(torch.tensor(x, device=device, dtype=dtype))
        return lgamma_cache[x]

    for L in range(L_max):
        for K in range(K_max):
            for J in range(J_max):
                A_block = torch.zeros_like(c8)
                for l in range(block_size * L, block_size * (L + 1)):
                    for k in range(block_size * K, block_size * (K + 1)):
                        for j in range(block_size * J, block_size * (J + 1)):
                            log_fact  = lgamma_cached(2*l+1) + lgamma_cached(2*k+1) + lgamma_cached(j+1)
                            log_gamma = (lgamma_cached(k + 0.5) + lgamma_cached(j + l + 0.5)
                                         - lgamma_cached(j + l + k + 1.5))
                            log_power = (2*(l+k)*torch.log(kappa) + j*torch.log(beta)
                                         + 2*l*torch.log(nu2) + 2*k*torch.log(nu3))
                            F0 = hyp0f1_torch(j+l+k+1.5, (kappa**2 * nu1**2)/4).clamp(1e-8, 1e8)
                            F2 = hyp2f1_torch(-j, k+0.5, 0.5-j-l, -eta).clamp(-1e8, 1e8)
                            A_block += torch.exp(log_power - log_fact + log_gamma) * F0 * F2
                c8 += A_block

    c8 *= 2 * torch.sqrt(torch.tensor(torch.pi, device=device))
    return c8
