# Theoretical Background

## FDS and PSD in fdscore

In `fdscore`, a **Fatigue Damage Spectrum (FDS)** is represented by
`FDSResult(f, damage)`, where `f` is the oscillator natural-frequency grid and
`damage` contains the predicted accumulated damage for each SDOF oscillator.
A **Power Spectral Density (PSD)** is represented by `PSDResult(f, psd)` as a
one-sided acceleration PSD defined over frequency.

The main workflows in the package use an SDOF oscillator bank defined by
`SDOFParams` together with an S-N model defined by `SNParams`. Damping is
consistently written internally as

```{math}
\zeta = \frac{1}{2Q}
```

which is the relationship used throughout the implementation when converting
the quality factor `q` to damping ratio.

## Time-Domain FDS

`compute_fds_time(...)` reconstructs each oscillator response in the frequency
domain using an FFT transfer matrix and then applies rainflow counting and
Miner damage accumulation. The implementation in `fdscore.rainflow_damage`
works on reversal points and sums full-cycle and half-cycle contributions in
the form

```{math}
D = \sum_j \phi_j \frac{S_j^k}{C}
```

where `\phi_j` is the cycle fraction (`0.5` or `1.0` in the code), `S_j` is
the effective cycle load, `k` is the S-N slope exponent, and
`C = N_{ref} S_{ref}^k`.

The `compute_fds_time(...)` documentation also makes explicit that, for fixed
`k`, the absolute damage level scales globally as

```{math}
\frac{p_{scale}^k}{N_{ref} S_{ref}^k}
```

which changes the magnitude of `damage(f)` without changing its relative
shape.

## Spectral FDS via Dirlik

`compute_fds_spectral_psd(...)` builds the spectral response of each
oscillator from a base-acceleration PSD. Its implementation explicitly uses

```{math}
P_{resp}(f; f_0) = p_{scale}^2 \, |H(f; f_0)|^2 \, P_{base}(f)
```

and

```{math}
damage(f_0) = \frac{duration_s}{life(f_0)}
```

where `life(f_0)` is obtained from the internal Dirlik implementation. The companion route
`compute_fds_spectral_time(...)` first estimates the PSD with Welch
(`compute_psd_welch(...)`) and then applies the same spectral fatigue
evaluation.

The implementation explicitly notes that Dirlik is a spectral fatigue
approximation, not the same algorithm as counting rainflow cycles directly on
a finite time-history realization. Agreement between the two routes is
therefore approximate rather than exact.

## Henderson-Piersol Closed-Form Inversion

The module `fdscore.inverse_closed_form` documents the closed-form derivation
used to invert an FDS into an equivalent acceleration PSD. The function
`compute_damage_to_dp_factor(...)` presents the damage expression

```{math}
D = \left(\frac{\nu_0 T}{C}\right)\left(\sqrt{2}\,\sigma_S\right)^b
\Gamma\left(1 + \frac{b}{2}\right)
```

and the approximate relationship between the stress standard deviation and the
input acceleration PSD

```{math}
\sigma_S \approx p_{scale}\sqrt{\frac{G_{aa}(f_n)}{16 \pi f_n \zeta}}
```

under the assumptions of narrowband Gaussian response, stationarity, SDOF
behavior, and light damping.

Those assumptions are central to the interpretation of the inversion:

- the excitation is treated as stationary Gaussian;
- the response is modeled as linear and SDOF-dominated;
- light damping is assumed, typically `zeta < 0.1`;
- the response peaks follow the Rayleigh law associated with narrowband
  Gaussian response;
- the input PSD is treated as locally flat across the oscillator half-power
  bandwidth.

In practical terms, the closed-form route returns the PSD of an equivalent
stationary Gaussian environment that would reproduce the same damage under
those assumptions. This is especially important when the input FDS was
originally computed from time-domain rainflow on a real signal rather than
from a purely spectral Gaussian model.

From there, the code defines the **Damage Potential (DP)** as

```{math}
DP(f_n) = f_n T \left[\frac{G_{aa}(f_n)}{f_n \zeta}\right]^{b/2}
```

and implements its algebraic inverse as

```{math}
G_{aa}(f_n) = f_n \zeta \left[\frac{DP(f_n)}{f_n T}\right]^{2/b}
```

`invert_fds_closed_form(...)` first converts damage to `DP(f)` using the
proportionality factor `K` such that `Damage = K * DP`, and then applies the
inverse above. The implementation restricts this inversion to `metric="pv"`,
using pseudo-velocity as a numerically convenient proxy for relative velocity
at resonance.

## Iterative Inversion Engines

The package also implements two iterative engines that synthesize a PSD to
match a target FDS:

- `invert_fds_iterative_spectral(...)` uses `compute_fds_spectral_psd(...)`
  as its internal predictor and updates the PSD multiplicatively from an
  influence matrix derived from the SDOF transfer function in the spectral
  domain.
- `invert_fds_iterative_time(...)` synthesizes Gaussian time histories from a
  candidate PSD with `synthesize_time_from_psd(...)`, evaluates the resulting
  FDS with `compute_fds_time(...)`, and corrects the PSD from the mismatch
  between predicted and target damage.

In both cases, the output remains a **one-sided acceleration PSD** defined on
the user-provided `f_psd_hz` grid, while the returned metadata records the
convergence history and the parameters effectively used by each engine.

For a practical comparison of when the closed-form and iterative engines are
methodologically aligned with the chosen FDS path, see
[workflows/inversion.md](workflows/inversion.md).
