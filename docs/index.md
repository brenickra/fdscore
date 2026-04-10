# fdscore

**fdscore** is a Python library for **Fatigue Damage Spectrum (FDS)**
computation and FDS-to-**Power Spectral Density (PSD)** inversion.

It implements the Henderson-Piersol closed-form inversion method, supports
multiple inversion engines (closed form, spectral iterative, and time-domain
iterative), and provides utilities for rainflow counting, Dirlik-based
spectral fatigue estimation, and Miner-rule damage accumulation.

---

## Documentation Guide

Use this site as the canonical documentation for installation, workflow
guidance, API reference, and theory. The public API documented here matches
the stable namespace exposed by `fdscore`.

The repository also ships a `CONTRACTS.md` document that defines the public
engineering contracts and compatibility rules used across the main release
workflows.

## Contents

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
quickstart
examples
concepts
```

```{toctree}
:maxdepth: 2
:caption: User Guide

workflows/fds
workflows/ers
workflows/shock
workflows/inversion
workflows/metrics
compatibility
assumptions-and-limits
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
theory
references
```
