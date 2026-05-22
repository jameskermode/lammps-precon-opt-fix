# Standalone `Exp` Preconditioner for LAMMPS — Spec with Stage-by-Stage ASE Validation

## Goal and scope

Deploy the Packwood `Exp` preconditioner (Packwood et al., JCP 2016) as a standalone capability for geometry optimisation in LAMMPS, working across arbitrary interatomic potentials (classical, EAM, Tersoff, MLIPs, foundation models). This is the validated, universal product: the chemistry-aware tables (v1-v4) were shown not to beat `Exp` and to inflate the condition number, so `Exp` is what ships. The preconditioner *form* is settled; the engineering effort goes into (1) scalable sparse assembly, (2) the sparse linear solve, and (3) variable-cell handling — validated stage-by-stage against ASE's reference `Exp` implementation.

Out of scope (deliberately): chemistry-aware stiffness tables, angle/FF terms (v5, separate), curvature/Hessian-based extensions (separate, would build on the Swinburne `H·v` operator). The structural design should leave a clean seam for FF angle terms and an `H·v`-based curvature substrate later, but neither is built here.

## The `Exp` preconditioner — what must be reproduced

`Exp` builds a sparse SPD matrix `P` (size `d·N × d·N`, `d=3` plus cell DOF for variable cell) with, for each pair `(i,j)` within cutoff `r_cut`, an isotropic coupling

```
P_ij block = -μ · exp(-A · (r_ij / r_NN - 1)) · I_d        (off-diagonal)
```

with diagonal blocks set so rows sum to the stabilisation term (graph-Laplacian structure), plus a stabilisation `c_stab` on the diagonal to keep `P` strictly positive definite. `r_NN` is the estimated nearest-neighbour distance, `A` is the decay constant (default 3.0 in ASE), and `μ` is the single global scale.

Each optimiser step solves `P · s = -g` (or applies `P⁻¹` to the gradient/step), where `g` is the force vector. This is the sparse solve that is the main engineering concern.

Reference: ASE `ase.optimize.precon` — `Exp` / `SparseCoeffPrecon`. `P` is a `scipy.sparse.csr_matrix`; ASE inverts it with scipy sparse linear algebra, falling back to **PyAMG algebraic-multigrid** for large systems (per ASE MR !258). `μ` is set per structure by `estimate_mu`, which costs two finite-difference force evaluations.

## The sparse-solve decision (resolved — read before implementing)

The concern about the sparse solve is well-founded but resolvable, and the key realisation removes most of the difficulty:

**`P` is SPD, so the scalable solve is conjugate gradient (CG), which needs only sparse matrix-vector products `P·v` — no factorisation, no external solver dependency.** `P` is a regularised graph-Laplacian-like operator: sparse, well-conditioned, few CG iterations to converge. This sidesteps the entire "which sparse direct solver / is it linkable in LAMMPS" question for the large-system path.

Adopt a **two-tier solver strategy**, mirroring what ASE does (scipy-direct → PyAMG-multigrid) but with a dependency-light default:

- **Small / medium systems (≲ a few thousand DOF): direct solve.** Eigen's `SimplicialLDLT` is the SPD-appropriate sparse direct solver. Eigen is **header-only with no dependencies beyond the C++ standard library**, so it is always available at compile time — no plugin to confirm, no link-time fragility. (Do NOT use Eigen's sparse QR — it is documented as inefficient. `SimplicialLDLT` is the SPD choice; `SimplicialLLT` also works.)
- **Large systems: conjugate gradient with a cheap preconditioner.** CG needs only `P·v` (sparse matvec), zero external dependency. Add a diagonal/Jacobi preconditioner to the CG (nearly free) for robustness. For ASE-equivalent scaling to very large systems, an algebraic-multigrid preconditioner on the CG (the role PyAMG plays in ASE) is the optional upgrade — but it is an *optimisation*, not a requirement: plain CG with Jacobi preconditioning on the SPD `P` will converge.

**On the assumed LAMMPS Eigen plugin:** I could NOT confirm that LAMMPS ships or links a dedicated Eigen plugin for linear algebra. Eigen's general sparse-solver capabilities are well documented, and Eigen is header-only so it can be vendored/included directly regardless. **Before relying on a LAMMPS-provided Eigen integration, confirm it exists in the target LAMMPS build** (check the build's installed packages / linked libraries). The design above does NOT require it — Eigen-as-header-only (vendored) for the direct branch, and CG-with-matvec (no external solver) for the iterative branch, are both available unconditionally. Treat any LAMMPS-Eigen plugin as a convenience if present, not a dependency.

Decision rule for tier selection: a DOF threshold (default ~5000, tunable) switches direct→iterative. Validate both tiers against ASE independently (Stage 4 below).

## Architecture

Recommended: a **Python-orchestrated** implementation first (LAMMPS as the force/energy engine via its Python module, `Exp` assembly and solve in Python/scipy), validated against ASE, then — only if performance requires — a C++ LAMMPS-native implementation. Rationale:

- The Python-orchestrated version can reuse ASE's exact `Exp` assembly code directly, making Stage 1-3 validation trivial (it *is* the ASE code, driven against LAMMPS forces).
- It immediately gives a working, validated capability for the system sizes where Python orchestration is acceptable.
- It de-risks the science (is `Exp`-on-LAMMPS-forces correct?) before investing in C++ performance work.
- The `LammpsImplicitDerivative` package (Maliyov/Grigorev/Swinburne) already drives LAMMPS from Python for exactly this kind of operator work and is a model for the orchestration / a place to share infrastructure.

The C++ LAMMPS-native version (a `fix` or `min_style`) is the eventual high-performance target for the largest systems and tightest integration, but it should be built only after the Python version has validated the science and quantified where Python orchestration becomes the bottleneck.

This spec covers the Python-orchestrated version and its validation; it notes the C++ port as a later stage.

## Stage-by-stage implementation and validation against ASE

Each stage produces a checkable artifact validated against ASE's `Exp` before proceeding. The validation philosophy is the same disciplined approach used throughout the project: make each stage falsifiable, confirm parity before building on it.

### Stage 0 — reference harness

Set up the ASE reference: a set of test structures (reuse the validation set — Si slab, MgO supercells, LaAlO₃, a defected cell; plus a few sizes for scaling) and ASE's `Exp` preconditioned LBFGS as the gold standard. Record, for each structure: `r_NN`, `μ` (from `estimate_mu`), the assembled `P` matrix (save as sparse), the per-step solve outputs, force-eval counts, and final relaxed structure.

These ASE artifacts are the targets every subsequent stage is checked against.

### Stage 1 — neighbour list and `r_NN` parity

LAMMPS provides neighbour lists natively. Confirm that the pair list used for `P` assembly (within `r_cut`) matches ASE's, and that `r_NN` is estimated identically.

**Validation:** for each test structure, the set of pairs `(i, j, r_ij)` within `r_cut` from LAMMPS must match ASE's (same pairs, same distances to ~1e-10, accounting for minimum-image/PBC). `r_NN` must match. Mismatches here are usually PBC/minimum-image or cutoff-boundary edge cases — resolve before proceeding.

### Stage 2 — `P` matrix assembly parity

Assemble `P` from the LAMMPS-derived pair list using the `Exp` formula. For the Python-orchestrated version, reuse ASE's assembly directly on the LAMMPS pair list.

**Validation:** the assembled sparse `P` must match ASE's `P` for the same structure and same `μ`:
- Same sparsity pattern (same nonzero locations).
- `norm(P_lammps - P_ase) / norm(P_ase) < 1e-10`.
- `P` symmetric (`norm(P - Pᵀ) < 1e-12`) and SPD (`SimplicialLDLT`/Cholesky succeeds; smallest eigenvalue > 0, equal to the `c_stab` floor).

Check on a single-element system (MgO/Si) and a multi-element one (LaAlO₃) — the multi-element case exercises any species-dependent pathway and is where the validation project found canonicalisation-type bugs.

### Stage 3 — `μ` estimation parity

Reproduce `estimate_mu` (the two-force-call finite-difference estimate) against LAMMPS forces.

**Validation:** `μ_lammps` matches `μ_ase` to the tolerance set by the finite-difference step (should agree to several digits). Confirm the two probe force evaluations are computed identically (same displacement, same FD convention). Note: this is the *universal* μ-setter — the autograd-HVP μ shortcut (`mu_replacement_spec.md`) is foundation-model-only and does NOT generalise across arbitrary LAMMPS potentials, so the finite-difference probe is what the LAMMPS deployment uses. (A future curvature-based universal μ could come from the Swinburne `H·v` operator — out of scope here, seam noted.)

Test μ estimation with at least two different potentials (e.g. a classical EAM and an MLIP) to confirm the probe is genuinely potential-agnostic.

### Stage 4 — the sparse solve parity (the main concern)

Validate the `P·s = -g` solve, both tiers, against ASE.

**4a — direct solve (small/medium).** Solve `P s = b` with Eigen `SimplicialLDLT` (or scipy in the Python version) for a known `b` (e.g. the initial force vector). Compare `s` against ASE's solve of the same system:
- `norm(s_lammps - s_ase) / norm(s_ase) < 1e-8`.
- Verify the residual `norm(P s - b) / norm(b)` is at solver tolerance.

**4b — iterative solve (large).** Solve the same system with CG (+ Jacobi preconditioner) and confirm it converges to the same `s` as the direct solve:
- `norm(s_cg - s_direct) / norm(s_direct) < CG_tol` (set CG tolerance e.g. 1e-8).
- Record CG iteration count vs system size — confirm it stays low (well-conditioned SPD), validating that CG is the right scalable choice.
- If an AMG-preconditioned CG is implemented, confirm it matches and reduces iteration count further.

**4c — tier-switch consistency.** At the DOF threshold, confirm direct and iterative tiers give the same `s` (within CG tolerance), so the switch is seamless.

This stage directly addresses the sparse-solve concern: 4a confirms correctness against an exact factorisation, 4b confirms the dependency-light CG path is correct and scalable, 4c confirms the two agree.

### Stage 5 — full preconditioned relaxation parity (fixed cell)

Run full preconditioned LBFGS relaxations on the fixed-cell test structures, using LAMMPS forces + the `Exp` solve, and compare against ASE's preconditioned LBFGS.

**Validation:**
- Final relaxed energies match to optimiser tolerance.
- Final structures match (RMSD within tolerance).
- Force-evaluation counts match ASE closely (small differences from solve-tolerance/line-search details acceptable; large differences indicate a bug). The validation report's numbers (e.g. Si slab ~33-40, LaAlO₃ ~56 for Exp) are the expected ballpark.
- Convergence robustness matches: where ASE `Exp` converges, the LAMMPS version converges.

### Stage 6 — variable-cell relaxation (the bug-prone path)

Variable-cell optimisation adds cell degrees of freedom and the cell-metric scaling `μ_c`. This path had a confirmed bug in the original validation (the `r_cut=None` variable-cell bug in `runtime.py`), so treat it with extra care.

**Validation:**
- The cell DOF are added to `P` with the correct `μ_c` (match ASE's variable-cell `Exp` assembly).
- Confirm `r_cut` and `r_NN` are resolved before `μ`/`μ_c` estimation (the locus of the known bug).
- Full variable-cell relaxation on a test case (γ-Al₂O₃ was the variable-cell loss case; use it) matches ASE: final cell, final structure, energy, force-eval count.
- Verify on a case with significant cell change (not just internal relaxation) so the cell-DOF path is genuinely exercised.

### Stage 7 — scaling validation

Confirm the implementation scales as intended and the tier-switch behaves.

**Validation:**
- Run the rocksalt supercell series (MgO ×2-×5, larger if feasible) and confirm force-eval counts stay flat with size (the `Exp` size-scaling signature from the validation report — Exp 12→26 over ×2-×5), matching ASE where ASE can run.
- Confirm the CG iteration count and solve wall-time scale acceptably (near-linear in N for the sparse SPD system).
- Confirm assembly cost stays well below force-eval cost.
- Push to system sizes beyond what ASE/Python can handle to demonstrate the scalability that motivates the LAMMPS deployment (these have no ASE reference — validate internally via residuals and convergence).

### Stage 8 (later) — C++ LAMMPS-native port

Only after Stages 0-7 validate the Python-orchestrated version. Reimplement assembly + solve as a LAMMPS `fix`/`min_style` in C++ (Eigen header-only for the direct solve; a CG with sparse matvec for the iterative path). Validate the C++ version against the validated Python version stage-by-stage (same parity checks: assembly, solve, relaxation). The Python version is now the reference, since it is itself validated against ASE.

## Solver dependency summary

- **Default, always-available:** Eigen header-only (`SimplicialLDLT` direct, small/medium) + CG-with-Jacobi (matvec-only, large). No external solver library required, no LAMMPS-Eigen plugin required.
- **Optional upgrade:** AMG-preconditioned CG for very-large-system scaling (the ASE/PyAMG role). Implement only if Stage 7 shows plain CG iteration counts growing unacceptably.
- **To confirm, not to rely on:** whether the target LAMMPS build provides an Eigen integration that could be reused instead of vendoring Eigen. Convenience if present; design assumes it is not.

## Structural seams for later work (do not build now)

- **FF angle terms (v5):** the assembly stage should be structured so additional rank-1 coordinate contributions (angles) could be added to `P` later, following the Mones Exp+FF construction. Keep the pair-assembly modular.
- **Curvature / `H·v` substrate:** leave a clean interface where the solve operator could be swapped from the fixed `Exp` `P` to a curvature-based operator using the Swinburne `LammpsImplicitDerivative` `H·v` (universal across potentials, scalable). This is the route to the off-diagonal structure the validation identified as the real source of speedup — but it is a separate project; here, just don't architecturally preclude it.

## Why this is the right deployment

- `Exp` is the validated, universal preconditioner — beats nothing-preconditioner on robustness (the real win: LBFGS fails half of defected structures, Exp converges all), ties or beats the chemistry-aware tables, needs no per-potential parameterisation.
- The SPD structure makes the feared sparse solve tractable with zero hard external dependency: direct (Eigen header-only) for small, CG-matvec for large.
- Stage-by-stage ASE parity de-risks every component before the next is built.
- The Python-first architecture validates the science cheaply before C++ performance work.
- The design leaves clean seams for the two evidence-motivated extensions (FF angles, `H·v` curvature) without committing to them.

## Implementation effort estimate

- Stages 0-5 (Python-orchestrated, fixed cell, both solver tiers, ASE parity): ~1-2 weeks.
- Stage 6 (variable cell): ~few days, with care at the known bug locus.
- Stage 7 (scaling): ~few days.
- Stage 8 (C++ native port): separate, larger effort — scope after Python version proves out.
