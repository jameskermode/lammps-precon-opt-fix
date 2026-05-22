/* Stage 8b — `fix precon/exp`, the domain-decomposed Exp preconditioner.

   Owns the preconditioner P and provides the operations `min_style
   precon/lbfgs` needs. P is row-distributed: each MPI rank holds the rows for
   its owned atoms, with columns referencing owned + ghost atoms. The
   matrix-vector product P*v is a halo exchange (`comm->forward_comm`) followed
   by a purely local sparse matvec — the same communication pattern as a force
   computation — so the solve (Jacobi-preconditioned CG) domain-decomposes
   naturally. No external linear-algebra dependency.

     fix ID group precon/exp [r_cut <value>] [dump <prefix>] [trace]
*/
#ifndef LMP_FIX_PRECON_EXP_H
#define LMP_FIX_PRECON_EXP_H

#include "fix.h"

#include <string>
#include <vector>

namespace LAMMPS_NS {

class FixPreconExp : public Fix {
 public:
  FixPreconExp(class LAMMPS *, int, char **);

  int setmask() override;
  void init() override;
  void init_list(int, class NeighList *) override;
  void min_post_force(int) override;     // per-force-call convergence trace
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;

  // --- preconditioner interface (driven by MinPreconLBFGS) ---------------
  void set_params(double A, double c_stab);
  double compute_r_NN();                 // distributed; sets and returns r_NN
  void set_geometry(double r_NN, double r_cut);
  void assemble(double mu);              // build the r_cut pair list at mu
  void scale(double factor);             // P <- factor * P
  void matvec(const double *v, double *out);          // out = P v  (owned)
  int solve(const double *b, double *x);  // distributed Jacobi-CG -> iters
  void dump(const std::string &prefix);   // serial: square P -> .mtx + .json

  double r_NN() const { return r_NN_; }
  double r_cut() const { return r_cut_; }
  double mu() const { return mu_; }
  double mu_c() const { return mu_c_; }
  void set_mu_c(double v) { mu_c_ = v; }
  double user_r_cut() const { return user_r_cut_; }
  const std::string &dump_prefix() const { return dump_prefix_; }
  bool trace() const { return trace_; }   // emit per-iteration PRECON_TRACE
  int last_cg_iterations() const { return cg_iterations_; }

 private:
  double ddot(const double *a, const double *b, int n) const;  // MPI dot

  class NeighList *list_ = nullptr;
  double A_ = 3.0, c_stab_ = 0.1;
  double r_NN_ = -1.0, r_cut_ = -1.0, mu_ = 1.0, mu_c_ = 1.0;
  double user_r_cut_ = -1.0;
  std::string dump_prefix_;
  bool trace_ = false;
  int trace_count_ = 0;    // cumulative force evaluations (when trace_)
  double cg_rtol_ = 1.0e-10;
  int cg_iterations_ = 0;

  // assembled P in pair-list form: off-diagonal (pair_i_ owned, pair_j_
  // owned-or-ghost, pair_coeff_) plus the per-owned-atom diagonal scalar.
  std::vector<int> pair_i_, pair_j_;
  std::vector<double> pair_coeff_;
  std::vector<double> diag_;

  std::vector<double> comm_vec_;   // 3*nall halo-exchange buffer
  std::vector<double> cg_r_, cg_z_, cg_p_, cg_Ap_;  // CG work vectors
};

}  // namespace LAMMPS_NS

#endif
