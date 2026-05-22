/* Stage 8b — `min_style precon/lbfgs`, MPI-aware.

   Preconditioned LBFGS. All reductions (the two-loop dot products, the FD
   mu-probe sums) go through MPI_Allreduce, and the LBFGS history is reset on
   every neighbour-list rebuild (after which atoms may have migrated between
   ranks) — so the minimizer is correct under domain decomposition and
   degenerates to the serial case at one rank. The linear solve is delegated
   to `fix precon/exp`'s distributed Jacobi-CG.

   Variable-cell relaxation (Stage 8c): when a `fix box/relax` is present the
   minimizer also relaxes the extra global cell DOF (`nextra_global`/`fextra`/
   `hextra`).  The cell block of the preconditioner (`mu_c`) is on a very
   different scale from the atomic block (`P_pos`), so the cell DOF get their
   own separate LBFGS recursion (`H0 = 1/mu_c`) rather than being mixed into
   the atomic two-loop — a combined recursion gives ill-conditioned `y·s`
   dot products dominated by whichever block is larger.  The shared line
   search couples the two.
*/
#ifndef LMP_MIN_PRECON_LBFGS_H
#define LMP_MIN_PRECON_LBFGS_H

#include "min_linesearch.h"

#include <vector>

namespace LAMMPS_NS {

class MinPreconLBFGS : public MinLineSearch {
 public:
  MinPreconLBFGS(class LAMMPS *);

  void init() override;
  int iterate(int) override;

 private:
  void setup_precon();   // first-call: locate the fix, r_NN, mu, assemble
  void estimate_mu();    // distributed two-force-call finite-difference probe
  double ddot(const double *a, const double *b, int n) const;  // MPI dot

  class FixPreconExp *fix_ = nullptr;
  bool precon_ready_ = false;
  int memory_ = 100;           // LBFGS history length
  bigint last_ncalls_ = -1;    // neighbour-build counter (history-reset signal)

  // atomic-DOF LBFGS history (length nvec, per rank)
  std::vector<std::vector<double>> s_hist_, y_hist_;
  std::vector<double> rho_hist_;
  std::vector<double> grad_, g_prev_, x_prev_;
  // cell-DOF LBFGS history (length nextra_global, replicated)
  std::vector<std::vector<double>> sc_hist_, yc_hist_;
  std::vector<double> rhoc_hist_;
  std::vector<double> gradc_, gc_prev_;
  bool has_prev_ = false;
};

}  // namespace LAMMPS_NS

#endif
