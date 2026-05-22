"""lammps_precon — standalone Exp preconditioner for LAMMPS geometry optimisation.

Python-orchestrated implementation: LAMMPS (+ Symmetrix MACE pair style) is the
force engine, ASE's reference ``Exp`` preconditioner is the validation gold
standard. See ``spec.md`` for the stage-by-stage plan.
"""

__version__ = "0.1.0"
