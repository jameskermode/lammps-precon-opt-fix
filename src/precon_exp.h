/* ----------------------------------------------------------------------
   Preconditioner for geometry optimization

   Based on ASE Exp preconditioner
   J. R. Kermode et al., J. Chem. Phys. 144, 164109 (2016)
------------------------------------------------------------------------- */

#ifndef LMP_PRECON_EXP_H
#define LMP_PRECON_EXP_H

#include <Eigen/Sparse>
#include <vector>
#include <memory>

namespace LAMMPS_NS {

class LAMMPS;

class PreconExp {
 public:
  PreconExp(double r_cut = -1.0, double mu = -1.0, double A = 3.0,
            double c_stab = 0.1);
  ~PreconExp();

  // Build preconditioner matrix
  void make_precon(LAMMPS *lmp, int groupbit, const std::vector<int> &fixed_tags,
                   class NeighList *list);

  // Estimate mu parameter automatically
  double estimate_mu(LAMMPS *lmp, int groupbit);

  // Estimate nearest neighbor distance
  double estimate_r_NN(LAMMPS *lmp, int groupbit, class NeighList *list);

  // Solve P * x = rhs
  void solve(const double *rhs, double *solution, int n);

  // Apply preconditioner: result = P * vec
  void apply(const double *vec, double *result, int n);

  // Get dense matrix (for testing)
  void get_matrix_dense(double *mat, int n);

  // Public parameters
  double r_cut;      // Neighbor cutoff
  double r_NN;       // Nearest neighbor distance
  double mu;         // Energy scale
  double A;          // Exponential decay parameter
  double c_stab;     // Stabilization constant

 private:
  // Sparse matrix and solver
  Eigen::SparseMatrix<double, Eigen::RowMajor> P_;
  std::unique_ptr<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>> solver_;
  bool factorized_;

  // Neighbor list data
  struct NeighborPair {
    int tag_i, tag_j;
    double rij;
  };
  std::vector<NeighborPair> neighbors_;

  // Private methods
  void extract_neighbors_from_lammps(LAMMPS *lmp, int groupbit, class NeighList *list);
  void assemble_matrix(LAMMPS *lmp, int groupbit,
                       const std::vector<int> &fixed_atoms);
  double compute_coefficient(double r);
};

}  // namespace LAMMPS_NS

#endif
