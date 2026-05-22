/* Stage 8b — `fix precon/exp`, domain-decomposed. See fix_precon_exp.h. */
#include "fix_precon_exp.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "utils.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <utility>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

FixPreconExp::FixPreconExp(LAMMPS *lmp, int narg, char **arg)
    : Fix(lmp, narg, arg) {
  comm_forward = 3;  // 3 doubles/atom forward-communicated for the matvec
  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "r_cut") == 0) {
      if (iarg + 2 > narg)
        error->all(FLERR, "fix precon/exp: r_cut needs a value");
      user_r_cut_ = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "dump") == 0) {
      if (iarg + 2 > narg)
        error->all(FLERR, "fix precon/exp: dump needs a prefix");
      dump_prefix_ = arg[iarg + 1];
      iarg += 2;
    } else if (strcmp(arg[iarg], "trace") == 0) {
      trace_ = true;   // emit a per-iteration PRECON_TRACE line
      iarg += 1;
    } else {
      error->all(FLERR, "fix precon/exp: unknown keyword");
    }
  }
}

int FixPreconExp::setmask() { return 0; }

void FixPreconExp::init() {
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

void FixPreconExp::init_list(int /*id*/, NeighList *ptr) { list_ = ptr; }

void FixPreconExp::set_params(double A, double c_stab) {
  A_ = A;
  c_stab_ = c_stab;
}

void FixPreconExp::set_geometry(double r_NN, double r_cut) {
  r_NN_ = r_NN;
  r_cut_ = r_cut;
}

/* ----------------------------------------------------------------------
   forward-communication of the matvec vector to ghost atoms
------------------------------------------------------------------------- */

int FixPreconExp::pack_forward_comm(int n, int *list, double *buf, int, int *) {
  int m = 0;
  for (int k = 0; k < n; ++k) {
    const int a = list[k];
    buf[m++] = comm_vec_[3 * a];
    buf[m++] = comm_vec_[3 * a + 1];
    buf[m++] = comm_vec_[3 * a + 2];
  }
  return m;
}

void FixPreconExp::unpack_forward_comm(int n, int first, double *buf) {
  int m = 0;
  for (int k = 0; k < n; ++k) {
    const int a = first + k;
    comm_vec_[3 * a] = buf[m++];
    comm_vec_[3 * a + 1] = buf[m++];
    comm_vec_[3 * a + 2] = buf[m++];
  }
}

double FixPreconExp::ddot(const double *a, const double *b, int n) const {
  double local = 0.0;
  for (int k = 0; k < n; ++k) local += a[k] * b[k];
  double global = 0.0;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, world);
  return global;
}

/* ----------------------------------------------------------------------
   r_NN: max over all atoms of the per-atom nearest-neighbour distance
------------------------------------------------------------------------- */

double FixPreconExp::compute_r_NN() {
  if (!list_) error->all(FLERR, "fix precon/exp: neighbour list unavailable");
  const int inum = list_->inum;
  const int nlocal = atom->nlocal;
  int *ilist = list_->ilist;
  int *numneigh = list_->numneigh;
  int **firstneigh = list_->firstneigh;
  double **x = atom->x;
  tagint *tag = atom->tag;

  std::vector<double> nearest(nlocal, 1.0e300);
  for (int ii = 0; ii < inum; ++ii) {
    const int i = ilist[ii];
    if (i >= nlocal) continue;
    int *jl = firstneigh[i];
    const int jn = numneigh[i];
    for (int jj = 0; jj < jn; ++jj) {
      const int j = jl[jj] & NEIGHMASK;
      if (atom->map(tag[j]) == i) continue;  // skip self-image
      const double dx = x[i][0] - x[j][0];
      const double dy = x[i][1] - x[j][1];
      const double dz = x[i][2] - x[j][2];
      const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
      if (r > 0.0 && r < nearest[i]) nearest[i] = r;
    }
  }
  double local_max = 0.0;
  for (int i = 0; i < nlocal; ++i)
    if (nearest[i] < 1.0e299 && nearest[i] > local_max) local_max = nearest[i];
  double global_max = 0.0;
  MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, world);
  r_NN_ = global_max;
  return r_NN_;
}

/* ----------------------------------------------------------------------
   assemble P (pair-list form) at the given mu
------------------------------------------------------------------------- */

void FixPreconExp::assemble(double mu) {
  if (r_NN_ <= 0.0 || r_cut_ <= 0.0)
    error->all(FLERR, "fix precon/exp: geometry not set before assemble");
  mu_ = mu;
  const int inum = list_->inum;
  const int nlocal = atom->nlocal;
  int *ilist = list_->ilist;
  int *numneigh = list_->numneigh;
  int **firstneigh = list_->firstneigh;
  double **x = atom->x;
  tagint *tag = atom->tag;

  pair_i_.clear();
  pair_j_.clear();
  pair_coeff_.clear();
  diag_.assign(nlocal, 0.0);

  for (int ii = 0; ii < inum; ++ii) {
    const int i = ilist[ii];
    if (i >= nlocal) continue;
    int *jl = firstneigh[i];
    const int jn = numneigh[i];
    for (int jj = 0; jj < jn; ++jj) {
      const int j = jl[jj] & NEIGHMASK;
      if (atom->map(tag[j]) == i) continue;  // skip self-image (like ASE)
      const double dx = x[i][0] - x[j][0];
      const double dy = x[i][1] - x[j][1];
      const double dz = x[i][2] - x[j][2];
      const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
      if (r > r_cut_) continue;
      const double coeff = -mu * std::exp(-A_ * (r / r_NN_ - 1.0));
      pair_i_.push_back(i);
      pair_j_.push_back(j);
      pair_coeff_.push_back(coeff);
      diag_[i] += -coeff;  // graph-Laplacian row sum
    }
  }
  for (int i = 0; i < nlocal; ++i) diag_[i] += mu * c_stab_;  // stabilisation
}

void FixPreconExp::scale(double factor) {
  for (auto &c : pair_coeff_) c *= factor;
  for (auto &d : diag_) d *= factor;
  mu_ *= factor;
}

/* ----------------------------------------------------------------------
   distributed matrix-vector product  out = P v  (owned-atom vectors)
------------------------------------------------------------------------- */

void FixPreconExp::matvec(const double *v, double *out) {
  const int nlocal = atom->nlocal;
  const int nall = nlocal + atom->nghost;
  comm_vec_.assign(3 * nall, 0.0);
  for (int k = 0; k < 3 * nlocal; ++k) comm_vec_[k] = v[k];
  comm->forward_comm(this);  // halo exchange: fills the ghost entries of v

  for (int i = 0; i < nlocal; ++i)
    for (int d = 0; d < 3; ++d)
      out[3 * i + d] = diag_[i] * comm_vec_[3 * i + d];
  const std::size_t np = pair_i_.size();
  for (std::size_t k = 0; k < np; ++k) {
    const int i = pair_i_[k], j = pair_j_[k];
    const double c = pair_coeff_[k];
    for (int d = 0; d < 3; ++d) out[3 * i + d] += c * comm_vec_[3 * j + d];
  }
}

/* ----------------------------------------------------------------------
   distributed Jacobi-preconditioned conjugate gradient: solve P x = b
------------------------------------------------------------------------- */

int FixPreconExp::solve(const double *b, double *x) {
  const int nlocal = atom->nlocal;
  const int n = 3 * nlocal;
  cg_r_.assign(n, 0.0);
  cg_z_.assign(n, 0.0);
  cg_p_.assign(n, 0.0);
  cg_Ap_.assign(n, 0.0);

  for (int k = 0; k < n; ++k) {
    x[k] = 0.0;
    cg_r_[k] = b[k];
  }
  const double bnorm = std::sqrt(ddot(b, b, n));
  if (bnorm == 0.0) {
    cg_iterations_ = 0;
    return 0;
  }
  auto jacobi = [&](const std::vector<double> &r, std::vector<double> &z) {
    for (int i = 0; i < nlocal; ++i)
      for (int d = 0; d < 3; ++d) z[3 * i + d] = r[3 * i + d] / diag_[i];
  };
  jacobi(cg_r_, cg_z_);
  for (int k = 0; k < n; ++k) cg_p_[k] = cg_z_[k];
  double rz = ddot(cg_r_.data(), cg_z_.data(), n);

  const int maxit = 5000;
  int it = 0;
  for (; it < maxit; ++it) {
    matvec(cg_p_.data(), cg_Ap_.data());
    const double pAp = ddot(cg_p_.data(), cg_Ap_.data(), n);
    const double alpha = rz / pAp;
    for (int k = 0; k < n; ++k) {
      x[k] += alpha * cg_p_[k];
      cg_r_[k] -= alpha * cg_Ap_[k];
    }
    const double rnorm = std::sqrt(ddot(cg_r_.data(), cg_r_.data(), n));
    if (rnorm <= cg_rtol_ * bnorm) {
      ++it;
      break;
    }
    jacobi(cg_r_, cg_z_);
    const double rz_new = ddot(cg_r_.data(), cg_z_.data(), n);
    const double beta = rz_new / rz;
    for (int k = 0; k < n; ++k) cg_p_[k] = cg_z_[k] + beta * cg_p_[k];
    rz = rz_new;
  }
  cg_iterations_ = it;
  return it;
}

/* ----------------------------------------------------------------------
   dump P (serial only) — fold ghost columns back to owned atoms so the
   result is the square nlocal*3 matrix the Python parity check expects
------------------------------------------------------------------------- */

void FixPreconExp::dump(const std::string &prefix) {
  if (comm->nprocs > 1) return;  // serial-only parity hook
  const int nlocal = atom->nlocal;
  tagint *tag = atom->tag;
  std::map<std::pair<int, int>, double> sq;
  for (std::size_t k = 0; k < pair_i_.size(); ++k) {
    const int i = pair_i_[k];
    const int J = atom->map(tag[pair_j_[k]]);
    for (int d = 0; d < 3; ++d) sq[{3 * i + d, 3 * J + d}] += pair_coeff_[k];
  }
  for (int i = 0; i < nlocal; ++i)
    for (int d = 0; d < 3; ++d) sq[{3 * i + d, 3 * i + d}] += diag_[i];

  const int ndof = 3 * nlocal;
  std::ofstream mtx(prefix + ".mtx");
  mtx.precision(17);
  mtx << "%%MatrixMarket matrix coordinate real general\n";
  mtx << ndof << ' ' << ndof << ' ' << sq.size() << '\n';
  for (const auto &kv : sq)
    mtx << (kv.first.first + 1) << ' ' << (kv.first.second + 1) << ' '
        << kv.second << '\n';
  mtx.close();

  std::ofstream js(prefix + ".json");
  js.precision(17);
  js << "{\"r_NN\": " << r_NN_ << ", \"r_cut\": " << r_cut_
     << ", \"mu\": " << mu_ << ", \"mu_c\": " << mu_c_
     << ", \"n_dof\": " << ndof << ", \"nnz\": " << sq.size() << "}\n";
  js.close();
}
