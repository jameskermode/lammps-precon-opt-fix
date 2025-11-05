/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Implementation based on ASE PreconLBFGS
   J. R. Kermode et al., J. Chem. Phys. 144, 164109 (2016)
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(precon_lbfgs,FixPreconLBFGS);
// clang-format on
#else

#ifndef LMP_FIX_PRECON_LBFGS_H
#define LMP_FIX_PRECON_LBFGS_H

#include "fix.h"
#include "precon_exp.h"
#include <vector>
#include <deque>
#include <string>

namespace LAMMPS_NS {

class FixPreconLBFGS : public Fix {
 public:
  FixPreconLBFGS(class LAMMPS *, int, char **);
  ~FixPreconLBFGS() override;

  int setmask() override;
  void init() override;
  void setup(int) override;
  void min_setup(int) override;
  void min_step(double, double *) override;
  void min_post_force(int) override;
  double min_energy(double *) override;
  void min_store() override;
  void min_clearstore() override;
  void min_pushstore() override;
  int min_reset_ref() override;
  double max_alpha(double *) override;
  int min_dof() override;
  void init_list(int, class NeighList *) override;

 private:
  // Neighbor list
  class NeighList *list;
  // LBFGS parameters
  int memory_;                      // History size (default: 100)
  double alpha_;                    // Initial Hessian guess 1/alpha (default: 70.0)
  double H0_;                       // 1/alpha
  double maxstep_;                  // Max single atom displacement (default: 0.04)
  double fmax_;                     // Force convergence criterion

  // Line search parameters
  bool use_armijo_;                 // Use Armijo (true) or Wolfe (false)
  double c1_;                       // Armijo parameter (default: 0.23)
  double c2_;                       // Wolfe parameter (default: 0.46)
  double a_min_;                    // Min line search step

  // Preconditioner parameters
  bool use_precon_;                 // Use preconditioner or identity
  double precon_r_cut_;             // Preconditioner cutoff
  double precon_mu_;                // Energy scale (-1 = auto)
  double precon_A_;                 // Exponential decay (default: 3.0)
  double precon_c_stab_;            // Stabilization (default: 0.1)

  // LBFGS history
  std::deque<std::vector<double>> s_;  // Position differences
  std::deque<std::vector<double>> y_;  // Gradient differences
  std::deque<double> rho_;             // 1/(y·s)

  // Current state
  std::vector<double> x0_;          // Previous positions (flat)
  std::vector<double> g0_;          // Previous gradient (flat)
  double e0_;                       // Previous energy
  double e1_;                       // Trial energy
  std::vector<double> p_;           // Search direction (flat)
  double alpha_k_;                  // Line search step size

  // Optimizer state
  int iteration_;
  bool reset_flag_;                 // Reset Hessian flag
  int natoms_total_;                // Total atoms in group
  int ndof_;                        // Degrees of freedom (3 * natoms)
  int linesearch_debug_;            // Line search debug level (0=off, 1=on)

  // Preconditioner
  PreconExp *precon_;

  // Energy computation
  class Compute *pe_compute_;

  // File I/O
  FILE *logfile_;
  std::string logfilename_;

  // Internal methods
  void parse_arguments(int narg, char **arg);
  void allocate_arrays();
  void reset_hessian();
  void compute_search_direction(const double *forces);
  void lbfgs_two_loop(const double *gradient, double *direction);
  double armijo_line_search(const double *x, const double *p,
                           const double *g, double e_current);
  void update_history(const double *x, const double *g);
  void get_positions_flat(double *x) const;
  void set_positions_flat(const double *x);
  void get_forces_flat(double *f) const;
  double compute_energy();
  double compute_trial_energy();  // Compute energy with force recomputation
  double compute_fmax(const double *forces) const;
  void write_log(double energy, double fmax);

  // Constraint detection
  std::vector<int> detect_fixed_atoms() const;

  // Utility
  double dot_product(const std::vector<double> &a, const std::vector<double> &b) const;
  double vector_norm(const std::vector<double> &v) const;
  double vector_max_abs(const std::vector<double> &v) const;
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal fix precon_lbfgs command

Self-explanatory. Check the input script syntax and compare to the
documentation for the command.

E: Fix precon_lbfgs requires atom map

The fix requires atoms to have tags/IDs and an atom map for the
preconditioner to work correctly.

E: LAMMPS neighbor cutoff too small for preconditioner

The pair_style cutoff must be >= preconditioner r_cut. Increase
pair_style cutoff or decrease precon r_cut parameter.

E: Fix precon_lbfgs failed to converge

The optimization did not converge within the maximum number of steps.

*/
