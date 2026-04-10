# Compatibility Rules

`fdscore` uses structured compatibility metadata to make algebra and inversion
operations explicit and safe.

## `meta["compat"]`

Most public result objects carry a compatibility signature inside
`meta["compat"]`. This signature is not an incidental implementation detail;
it is part of the engineering contract used to protect composition workflows.

## FDS Algebra Compatibility

Operations such as `sum_fds(...)` and `scale_fds(...)` work on spectra defined
on the oscillator grid. For that reason, they require:

- the same response metric;
- the same damping level;
- the same fatigue semantics (`SNParams`, `p_scale`, and FDS kind);
- the same oscillator frequency grid.

If those conditions do not hold, the operation is rejected rather than
regridding or silently combining incompatible spectra.

## Inversion Compatibility

Inversion uses compatibility differently.

For `invert_fds_closed_form(...)` and the iterative inversion engines, the
damage semantics must match, but the PSD grid is a separate object and does not
need to match the target FDS oscillator grid.

This distinction is intentional:

- FDS algebra combines spectra already defined on the same oscillator axis.
- inversion solves for a separate PSD representation on its own frequency
  axis.

## ERS and Shock Compatibility

ERS envelope composition requires compatible ERS semantics, including metric,
damping, peak mode, and oscillator grid.

Shock envelope helpers add one more layer: they also require compatible shock
kind, so SRS and PVSS are not mixed accidentally.

## Public Contract Scope

The generated API reference intentionally focuses on the stable public
namespace exposed by `fdscore`. The complete contract language used to define
those guarantees is maintained separately in `CONTRACTS.md` at the repository
root.
