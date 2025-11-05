/* ----------------------------------------------------------------------
   Preconditioner implementation

   Based on ASE Exp preconditioner
------------------------------------------------------------------------- */

#include "precon_exp.h"
#include "lammps.h"
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "error.h"
#include "input.h"
#include "update.h"
#include "force.h"
#include "pair.h"

#include <cmath>
#include <algorithm>
#include <map>
#include <set>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PreconExp::PreconExp(double r_cut_, double mu_, double A_, double c_stab_) :
  r_cut(r_cut_), r_NN(-1.0), mu(mu_), A(A_), c_stab(c_stab_),
  factorized_(false)
{
}

/* ---------------------------------------------------------------------- */

PreconExp::~PreconExp()
{
}

/* ---------------------------------------------------------------------- */

void PreconExp::make_precon(LAMMPS *lmp, int groupbit, const std::vector<int> &fixed_tags,
                            NeighList *list)
{
  // Estimate r_NN if not set
  if (r_NN < 0.0) {
    r_NN = estimate_r_NN(lmp, groupbit, list);
  }

  // Set cutoff if not specified
  if (r_cut < 0.0) {
    r_cut = 2.0 * r_NN;
  }

  // Estimate mu if not set
  if (mu < 0.0) {
    mu = estimate_mu(lmp, groupbit);
  }

  // Extract neighbors from LAMMPS
  extract_neighbors_from_lammps(lmp, groupbit, list);

  // Assemble sparse matrix with fixed atoms
  assemble_matrix(lmp, groupbit, fixed_tags);

  // Factorize for solving
  solver_ = std::make_unique<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>>();
  solver_->compute(P_);

  if (solver_->info() != Eigen::Success) {
    lmp->error->all(FLERR,"Preconditioner factorization failed");
  }

  factorized_ = true;
}

/* ---------------------------------------------------------------------- */

double PreconExp::estimate_mu(LAMMPS *lmp, int groupbit)
{
  /* Estimate optimal mu parameter using sine-based perturbation
   *
   * Solves: [dE(p+v) - dE(p)] · v = mu <P1 v, v>
   * where v(x,y,z) = H [sin(x/Lx), sin(y/Ly), sin(z/Lz)]
   * and P1 is preconditioner with mu=1
   *
   * Based on ASE PreconLBFGS estimate_mu() method
   */

  Atom *atom = lmp->atom;
  Comm *comm = lmp->comm;
  int *mask = atom->mask;
  double **x = atom->x;
  double **f = atom->f;
  tagint *tag = atom->tag;

  // Count local atoms in group
  int nlocal_group = 0;
  for (int i = 0; i < atom->nlocal; i++) {
    if (mask[i] & groupbit) nlocal_group++;
  }

  // Get total atom count
  int natoms_total = 0;
  MPI_Allreduce(&nlocal_group, &natoms_total, 1, MPI_INT, MPI_SUM, lmp->world);

  // Gather all positions to compute bounding box
  int nprocs = comm->nprocs;
  std::vector<int> recvcounts(nprocs);
  MPI_Allgather(&nlocal_group, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, lmp->world);

  std::vector<int> displs(nprocs);
  displs[0] = 0;
  for (int i = 1; i < nprocs; i++) {
    displs[i] = displs[i-1] + recvcounts[i-1];
  }

  // Gather local positions
  std::vector<double> pos_local(nlocal_group * 3);
  int idx = 0;
  for (int i = 0; i < atom->nlocal; i++) {
    if (mask[i] & groupbit) {
      pos_local[idx++] = x[i][0];
      pos_local[idx++] = x[i][1];
      pos_local[idx++] = x[i][2];
    }
  }

  // Adjust recvcounts and displs for 3D positions
  for (int i = 0; i < nprocs; i++) {
    recvcounts[i] *= 3;
    displs[i] *= 3;
  }

  // Gather all positions
  std::vector<double> pos_all(natoms_total * 3);
  MPI_Allgatherv(pos_local.data(), nlocal_group * 3, MPI_DOUBLE,
                 pos_all.data(), recvcounts.data(), displs.data(),
                 MPI_DOUBLE, lmp->world);

  // Compute bounding box
  double Lx = -1e10, Ly = -1e10, Lz = -1e10;
  double min_x = 1e10, min_y = 1e10, min_z = 1e10;

  for (int i = 0; i < natoms_total; i++) {
    double px = pos_all[3*i + 0];
    double py = pos_all[3*i + 1];
    double pz = pos_all[3*i + 2];

    if (px > Lx) Lx = px;
    if (py > Ly) Ly = py;
    if (pz > Lz) Lz = pz;

    if (px < min_x) min_x = px;
    if (py < min_y) min_y = py;
    if (pz < min_z) min_z = pz;
  }

  Lx = Lx - min_x;
  Ly = Ly - min_y;
  Lz = Lz - min_z;

  // Deformation matrix H = 1e-2 * r_NN * I
  double H = 1e-2 * r_NN;

  // Create perturbation vector: v = H * [sin(x/Lx), sin(y/Ly), sin(z/Lz)]
  std::vector<double> v_all(natoms_total * 3);

  for (int i = 0; i < natoms_total; i++) {
    double px = pos_all[3*i + 0];
    double py = pos_all[3*i + 1];
    double pz = pos_all[3*i + 2];

    // Handle zero dimensions
    v_all[3*i + 0] = (Lx > 1e-6) ? H * std::sin(px / Lx) : 0.0;
    v_all[3*i + 1] = (Ly > 1e-6) ? H * std::sin(py / Ly) : 0.0;
    v_all[3*i + 2] = (Lz > 1e-6) ? H * std::sin(pz / Lz) : 0.0;
  }

  // Get current forces (dE_p = -f)
  std::vector<double> dE_p_local(nlocal_group * 3);
  idx = 0;
  for (int i = 0; i < atom->nlocal; i++) {
    if (mask[i] & groupbit) {
      dE_p_local[idx++] = -f[i][0];
      dE_p_local[idx++] = -f[i][1];
      dE_p_local[idx++] = -f[i][2];
    }
  }

  // Gather current forces
  std::vector<double> dE_p(natoms_total * 3);
  MPI_Allgatherv(dE_p_local.data(), nlocal_group * 3, MPI_DOUBLE,
                 dE_p.data(), recvcounts.data(), displs.data(),
                 MPI_DOUBLE, lmp->world);

  // Save current positions and apply perturbation
  std::vector<double> x_save(atom->nlocal * 3);
  for (int i = 0; i < atom->nlocal; i++) {
    x_save[3*i + 0] = x[i][0];
    x_save[3*i + 1] = x[i][1];
    x_save[3*i + 2] = x[i][2];
  }

  // Build tag-to-index mapping for local atoms
  std::map<tagint, int> tag_to_local;
  idx = 0;
  for (int i = 0; i < atom->nlocal; i++) {
    if (mask[i] & groupbit) {
      tag_to_local[tag[i]] = i;
      idx++;
    }
  }

  // Gather all tags
  std::vector<tagint> tags_local(nlocal_group);
  idx = 0;
  for (int i = 0; i < atom->nlocal; i++) {
    if (mask[i] & groupbit) {
      tags_local[idx++] = tag[i];
    }
  }

  // Reset recvcounts for tags
  for (int i = 0; i < nprocs; i++) {
    recvcounts[i] /= 3;
    displs[i] /= 3;
  }

  std::vector<tagint> tags_all(natoms_total);
  MPI_Allgatherv(tags_local.data(), nlocal_group, MPI_LMP_TAGINT,
                 tags_all.data(), recvcounts.data(), displs.data(),
                 MPI_LMP_TAGINT, lmp->world);

  // Apply perturbation to local atoms
  for (int i = 0; i < natoms_total; i++) {
    tagint t = tags_all[i];
    auto it = tag_to_local.find(t);
    if (it != tag_to_local.end()) {
      int local_idx = it->second;
      x[local_idx][0] += v_all[3*i + 0];
      x[local_idx][1] += v_all[3*i + 1];
      x[local_idx][2] += v_all[3*i + 2];
    }
  }

  // Forward communicate positions
  comm->forward_comm();

  // Recompute forces using LAMMPS force evaluation
  // Clear forces
  int nall = atom->nlocal + atom->nghost;
  for (int i = 0; i < nall; i++) {
    f[i][0] = 0.0;
    f[i][1] = 0.0;
    f[i][2] = 0.0;
  }

  // Compute forces
  lmp->force->pair->compute(lmp->update->ntimestep, 1);

  // Reverse communicate forces
  comm->reverse_comm();

  // Get perturbed forces (dE_p_plus_v = -f)
  std::vector<double> dE_p_plus_v_local(nlocal_group * 3);
  idx = 0;
  for (int i = 0; i < atom->nlocal; i++) {
    if (mask[i] & groupbit) {
      dE_p_plus_v_local[idx++] = -f[i][0];
      dE_p_plus_v_local[idx++] = -f[i][1];
      dE_p_plus_v_local[idx++] = -f[i][2];
    }
  }

  // Gather perturbed forces
  for (int i = 0; i < nprocs; i++) {
    recvcounts[i] *= 3;
    displs[i] *= 3;
  }

  std::vector<double> dE_p_plus_v(natoms_total * 3);
  MPI_Allgatherv(dE_p_plus_v_local.data(), nlocal_group * 3, MPI_DOUBLE,
                 dE_p_plus_v.data(), recvcounts.data(), displs.data(),
                 MPI_DOUBLE, lmp->world);

  // Restore original positions
  for (int i = 0; i < atom->nlocal; i++) {
    x[i][0] = x_save[3*i + 0];
    x[i][1] = x_save[3*i + 1];
    x[i][2] = x_save[3*i + 2];
  }

  comm->forward_comm();

  // Recompute forces at original position
  // Clear forces
  for (int i = 0; i < nall; i++) {
    f[i][0] = 0.0;
    f[i][1] = 0.0;
    f[i][2] = 0.0;
  }

  // Compute forces
  lmp->force->pair->compute(lmp->update->ntimestep, 1);

  // Reverse communicate forces
  comm->reverse_comm();

  // Compute LHS = (dE_p_plus_v - dE_p) · v
  double LHS = 0.0;
  for (int i = 0; i < natoms_total * 3; i++) {
    LHS += (dE_p_plus_v[i] - dE_p[i]) * v_all[i];
  }

  // Build preconditioner with mu=1 to compute RHS
  double mu_save = mu;
  mu = 1.0;

  // Assemble matrix with mu=1 (need to pass empty fixed_tags)
  std::vector<int> no_fixed;
  assemble_matrix(lmp, groupbit, no_fixed);

  // Compute RHS = (P · v) · v
  Eigen::Map<const Eigen::VectorXd> v_eigen(v_all.data(), natoms_total * 3);
  Eigen::VectorXd Pv = P_ * v_eigen;

  double RHS = 0.0;
  for (int i = 0; i < natoms_total * 3; i++) {
    RHS += Pv[i] * v_all[i];
  }

  // Solve for mu
  double mu_est = LHS / RHS;

  // Clamp to minimum value of 1.0
  if (mu_est < 1.0) {
    if (comm->me == 0) {
      printf("  estimate_mu(): mu (%.3f) < 1.0, capping at mu=1.0\n", mu_est);
    }
    mu_est = 1.0;
  }

  // Restore original mu (to trigger rebuild later)
  mu = mu_save;

  return mu_est;
}

/* ---------------------------------------------------------------------- */

double PreconExp::estimate_r_NN(LAMMPS *lmp, int groupbit, NeighList *list)
{
  Atom *atom = lmp->atom;
  int *mask = atom->mask;
  double **x = atom->x;

  double local_min = 1e10;

  if (list) {
    // Use LAMMPS neighbor list
    int inum = list->inum;
    int *ilist = list->ilist;
    int *numneigh = list->numneigh;
    int **firstneigh = list->firstneigh;

    for (int ii = 0; ii < inum; ii++) {
      int i = ilist[ii];
      if (!(mask[i] & groupbit)) continue;

      int *jlist = firstneigh[i];
      int jnum = numneigh[i];

      for (int jj = 0; jj < jnum; jj++) {
        int j = jlist[jj];
        j &= NEIGHMASK;

        // Check if j is in group (for local atoms)
        if (j < atom->nlocal && !(mask[j] & groupbit)) continue;

        double dx = x[i][0] - x[j][0];
        double dy = x[i][1] - x[j][1];
        double dz = x[i][2] - x[j][2];
        double rij = std::sqrt(dx*dx + dy*dy + dz*dz);

        if (rij > 0.01 && rij < local_min) {
          local_min = rij;
        }
      }
    }
  } else {
    // Fallback: brute force if no neighbor list provided
    int nlocal = atom->nlocal;
    for (int i = 0; i < nlocal; i++) {
      if (!(mask[i] & groupbit)) continue;

      for (int j = i+1; j < nlocal + atom->nghost; j++) {
        if (j < nlocal && !(mask[j] & groupbit)) continue;

        double dx = x[i][0] - x[j][0];
        double dy = x[i][1] - x[j][1];
        double dz = x[i][2] - x[j][2];
        double rij = std::sqrt(dx*dx + dy*dy + dz*dz);

        if (rij > 0.01 && rij < local_min) {
          local_min = rij;
        }
      }
    }
  }

  // Find global minimum
  double global_min;
  MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, lmp->world);

  return global_min;
}

/* ---------------------------------------------------------------------- */

void PreconExp::solve(const double *rhs, double *solution, int n)
{
  if (!factorized_) {
    // Not initialized, just return identity
    for (int i = 0; i < n; i++) {
      solution[i] = rhs[i];
    }
    return;
  }

  Eigen::Map<const Eigen::VectorXd> b(rhs, n);
  Eigen::Map<Eigen::VectorXd> x(solution, n);

  x = solver_->solve(b);
}

/* ---------------------------------------------------------------------- */

void PreconExp::apply(const double *vec, double *result, int n)
{
  if (!factorized_) {
    // Not initialized, return identity
    for (int i = 0; i < n; i++) {
      result[i] = vec[i];
    }
    return;
  }

  Eigen::Map<const Eigen::VectorXd> v(vec, n);
  Eigen::Map<Eigen::VectorXd> r(result, n);

  r = P_ * v;
}

/* ---------------------------------------------------------------------- */

void PreconExp::get_matrix_dense(double *mat, int n)
{
  if (!factorized_) return;

  Eigen::MatrixXd dense = P_;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      mat[i*n + j] = dense(i, j);
    }
  }
}

/* ---------------------------------------------------------------------- */

void PreconExp::extract_neighbors_from_lammps(LAMMPS *lmp, int groupbit, NeighList *list)
{
  Atom *atom = lmp->atom;
  Comm *comm = lmp->comm;
  int *mask = atom->mask;
  double **x = atom->x;
  tagint *tag = atom->tag;

  neighbors_.clear();

  std::vector<NeighborPair> local_pairs;

  if (list) {
    // Use LAMMPS neighbor list
    int inum = list->inum;
    int *ilist = list->ilist;
    int *numneigh = list->numneigh;
    int **firstneigh = list->firstneigh;

    for (int ii = 0; ii < inum; ii++) {
      int i = ilist[ii];
      if (!(mask[i] & groupbit)) continue;

      int *jlist = firstneigh[i];
      int jnum = numneigh[i];

      for (int jj = 0; jj < jnum; jj++) {
        int j = jlist[jj];
        j &= NEIGHMASK;

        // Check if j is in group (for local atoms)
        if (j < atom->nlocal && !(mask[j] & groupbit)) continue;

        double dx = x[i][0] - x[j][0];
        double dy = x[i][1] - x[j][1];
        double dz = x[i][2] - x[j][2];
        double rij = std::sqrt(dx*dx + dy*dy + dz*dz);

        if (rij < r_cut) {
          NeighborPair pair;
          pair.tag_i = tag[i];
          pair.tag_j = tag[j];
          pair.rij = rij;
          local_pairs.push_back(pair);
        }
      }
    }
  } else {
    // Fallback: brute force if no neighbor list provided
    int nlocal = atom->nlocal;
    for (int i = 0; i < nlocal; i++) {
      if (!(mask[i] & groupbit)) continue;

      for (int j = i+1; j < nlocal + atom->nghost; j++) {
        if (j < nlocal && !(mask[j] & groupbit)) continue;

        double dx = x[i][0] - x[j][0];
        double dy = x[i][1] - x[j][1];
        double dz = x[i][2] - x[j][2];
        double rij = std::sqrt(dx*dx + dy*dy + dz*dz);

        if (rij < r_cut) {
          NeighborPair pair;
          pair.tag_i = tag[i];
          pair.tag_j = tag[j];
          pair.rij = rij;
          local_pairs.push_back(pair);
        }
      }
    }
  }

  // Gather neighbor pairs to all ranks via MPI
  int nlocal_pairs = local_pairs.size();
  int nprocs = comm->nprocs;

  // Gather counts from all ranks
  std::vector<int> recvcounts(nprocs);
  MPI_Allgather(&nlocal_pairs, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, lmp->world);

  // Compute total and displacements
  int total_pairs = 0;
  std::vector<int> displs(nprocs);
  for (int i = 0; i < nprocs; i++) {
    displs[i] = total_pairs;
    total_pairs += recvcounts[i];
  }

  // Pack local pairs into arrays for MPI
  std::vector<tagint> tags_i_local(nlocal_pairs);
  std::vector<tagint> tags_j_local(nlocal_pairs);
  std::vector<double> rij_local(nlocal_pairs);

  for (int i = 0; i < nlocal_pairs; i++) {
    tags_i_local[i] = local_pairs[i].tag_i;
    tags_j_local[i] = local_pairs[i].tag_j;
    rij_local[i] = local_pairs[i].rij;
  }

  // Gather all pairs
  std::vector<tagint> tags_i_all(total_pairs);
  std::vector<tagint> tags_j_all(total_pairs);
  std::vector<double> rij_all(total_pairs);

  MPI_Allgatherv(tags_i_local.data(), nlocal_pairs, MPI_LMP_TAGINT,
                 tags_i_all.data(), recvcounts.data(), displs.data(),
                 MPI_LMP_TAGINT, lmp->world);

  MPI_Allgatherv(tags_j_local.data(), nlocal_pairs, MPI_LMP_TAGINT,
                 tags_j_all.data(), recvcounts.data(), displs.data(),
                 MPI_LMP_TAGINT, lmp->world);

  MPI_Allgatherv(rij_local.data(), nlocal_pairs, MPI_DOUBLE,
                 rij_all.data(), recvcounts.data(), displs.data(),
                 MPI_DOUBLE, lmp->world);

  // Remove duplicates: keep only pairs where tag_i < tag_j
  // This eliminates both (i,j) and (j,i) duplicates
  std::set<std::pair<tagint, tagint>> unique_pairs;
  neighbors_.clear();

  for (int i = 0; i < total_pairs; i++) {
    tagint ti = tags_i_all[i];
    tagint tj = tags_j_all[i];

    // Ensure ti < tj for canonical ordering
    if (ti > tj) std::swap(ti, tj);

    // Check if we've seen this pair
    auto pair_key = std::make_pair(ti, tj);
    if (unique_pairs.find(pair_key) == unique_pairs.end()) {
      unique_pairs.insert(pair_key);

      NeighborPair pair;
      pair.tag_i = ti;
      pair.tag_j = tj;
      pair.rij = rij_all[i];
      neighbors_.push_back(pair);
    }
  }
}

/* ---------------------------------------------------------------------- */

void PreconExp::assemble_matrix(LAMMPS *lmp, int groupbit,
                                const std::vector<int> &fixed_tags)
{
  Atom *atom = lmp->atom;
  Comm *comm = lmp->comm;
  int *mask = atom->mask;
  tagint *tag = atom->tag;

  // Count local atoms in group
  int nlocal_group = 0;
  for (int i = 0; i < atom->nlocal; i++) {
    if (mask[i] & groupbit) nlocal_group++;
  }

  // Get total atom count via MPI
  int natoms_total = 0;
  MPI_Allreduce(&nlocal_group, &natoms_total, 1, MPI_INT, MPI_SUM, lmp->world);

  // Gather all tags to build global ordering
  int nprocs = comm->nprocs;
  std::vector<int> recvcounts(nprocs);
  MPI_Allgather(&nlocal_group, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, lmp->world);

  std::vector<int> displs(nprocs);
  displs[0] = 0;
  for (int i = 1; i < nprocs; i++) {
    displs[i] = displs[i-1] + recvcounts[i-1];
  }

  // Gather local tags
  std::vector<tagint> tags_local(nlocal_group);
  int idx = 0;
  for (int i = 0; i < atom->nlocal; i++) {
    if (mask[i] & groupbit) {
      tags_local[idx++] = tag[i];
    }
  }

  // Gather all tags
  std::vector<tagint> tags_all(natoms_total);
  MPI_Allgatherv(tags_local.data(), nlocal_group, MPI_LMP_TAGINT,
                 tags_all.data(), recvcounts.data(), displs.data(),
                 MPI_LMP_TAGINT, lmp->world);

  // Sort tags and build global tag-to-index mapping
  std::sort(tags_all.begin(), tags_all.end());
  std::map<tagint, int> tag_to_idx;
  for (int i = 0; i < natoms_total; i++) {
    tag_to_idx[tags_all[i]] = i;
  }

  // Build set of fixed tags for fast lookup
  std::set<int> fixed_set(fixed_tags.begin(), fixed_tags.end());

  // Initialize matrix
  int n = 3 * natoms_total;
  std::vector<Eigen::Triplet<double>> triplets;
  std::vector<double> diag(n, 0.0);

  // Assemble off-diagonal entries from neighbor pairs
  for (const auto &pair : neighbors_) {
    auto it_i = tag_to_idx.find(pair.tag_i);
    auto it_j = tag_to_idx.find(pair.tag_j);

    if (it_i == tag_to_idx.end() || it_j == tag_to_idx.end()) continue;

    int i = it_i->second;
    int j = it_j->second;

    double coeff = compute_coefficient(pair.rij);

    // Add 3x3 blocks for (i,j) and (j,i)
    for (int d = 0; d < 3; d++) {
      triplets.emplace_back(3*i+d, 3*j+d, coeff);
      triplets.emplace_back(3*j+d, 3*i+d, coeff);

      diag[3*i+d] -= coeff;
      diag[3*j+d] -= coeff;
    }
  }

  // Add diagonal entries
  for (int i = 0; i < natoms_total; i++) {
    tagint atom_tag = tags_all[i];
    bool is_fixed = (fixed_set.find(atom_tag) != fixed_set.end());

    for (int d = 0; d < 3; d++) {
      int idx_val = 3*i + d;

      if (is_fixed) {
        // Fixed atoms: set diagonal to 1, zero off-diagonals
        triplets.emplace_back(idx_val, idx_val, 1.0);
      } else {
        // Free atoms: use computed diagonal with stabilization
        double diag_val = diag[idx_val] + mu * c_stab;
        triplets.emplace_back(idx_val, idx_val, diag_val);
      }
    }
  }

  // Assemble sparse matrix
  P_.resize(n, n);
  P_.setFromTriplets(triplets.begin(), triplets.end());
}

/* ---------------------------------------------------------------------- */

double PreconExp::compute_coefficient(double r)
{
  // Exponential decay: -mu * exp(-A * (r/r_NN - 1))
  return -mu * std::exp(-A * (r / r_NN - 1.0));
}
