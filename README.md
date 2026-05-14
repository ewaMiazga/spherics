# spherics

PyTorch implementations of spherical probability distributions, designed for evaluation of spherical distributions on the unit sphere $S^2$.

## Installation

```bash
pip install spherics
```

Or from source:

```bash
git clone https://github.com/ewaMiazga/spherics.git
cd spherics
pip install -e .
```

## Distributions

| Function | Params / lobe | Description | Paper |
|---|---|---|---|
| `vMF` | 3 | von Mises–Fisher | [Fisher 1953](https://www.jstor.org/stable/3213566) |
| `spherical_gaussian` | 3 | Spherical Gaussian | [Fisher 1953](https://royalsocietypublishing.org/doi/10.1098/rspa.1953.0064) |
| `spherical_cauchy` | 3 | Spherical Cauchy | [Kato & McCullagh 2020](https://doi.org/10.3150/20-BEJ1222) |
| `spherical_beta` | 3 | Spherical Beta | [Trenkler 1996](https://econpapers.repec.org/article/eeecsdana/v_3a22_3ay_3a1996_3ai_3a5_3ap_3a568-569.htm) |
| `spherical_logistic` | 4 | Spherical Logistic | [Moghimbeygi & Golalizadeh 2020](https://doi.org/10.1007/s40304-018-00171-2) |
| `spherical_fb4` | 4 | Fisher–Bingham FB4 | [Kent 1982](https://www.jstor.org/stable/2335218) |
| `spherical_fb6` | 6 | Fisher–Bingham FB6 | [Yuan 2021](https://doi.org/10.1007/s00180-020-01023-w) |
| `spherical_fb8` | 8 | Fisher–Bingham FB8 | [Yuan 2021](https://doi.org/10.1007/s00180-020-01023-w) |
| `asg` | 5 | Anisotropic Spherical Gaussian | [Xu et al. 2013](https://doi.org/10.1145/2508363.2508386) |
| `nasg` | 5 | Normalized Anisotropic Spherical Gaussian | [Huang et al. 2024](https://doi.org/10.1145/3649310) |
| `nasg_gabor` | 6 | NASG with Gabor-style normalization | [missing]() |
| `ltc` | 4 | Linearly Transformed Cosine | [Heitz et al. 2016](https://doi.org/10.1145/2897824.2925895) |

## Usage

All functions share the same interface:

```python
import torch
from spherics import vMF, nasg, asg, ltc

# Sample directions on the unit sphere: shape (B, N, 3)
B, N = 4, 1000
dirs = torch.randn(B, N, 3)
dirs = dirs / dirs.norm(dim=-1, keepdim=True)

# Parameters: shape (B, n_params_per_lobe * n_lobes)
# Memory layout is lobe-first (interleaved):
#   [lobe0_p0, lobe0_p1, ..., lobe1_p0, lobe1_p1, ...]
params = torch.rand(B, 3)   # e.g. vMF with 1 lobe, 3 params

# Evaluate PDF: returns shape (B,)
pdf = vMF(dirs, params, n_gaussians=1, normalized=True)
```

## Parameter layout

Parameters are stored in **lobe-first interleaved** order. For a mixture with `K` lobes and `P` parameters per lobe:

```
params = [lobe0_p0, lobe0_p1, ..., lobe0_pP, lobe1_p0, ..., lobeK_pP]
```

## Running tests

```bash
pip install -e ".[test]"
pytest
```

## License

MIT
