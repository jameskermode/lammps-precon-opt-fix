/* ----------------------------------------------------------------------
   LAMMPS Plugin: Preconditioned LBFGS optimizer

   Based on ASE PreconLBFGS by J. R. Kermode et al.
   J. Chem. Phys. 144, 164109 (2016)
------------------------------------------------------------------------- */

#include "fix_precon_lbfgs.h"
#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "pair.h"
#include "update.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <map>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixPreconLBFGS::FixPreconLBFGS(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  memory_(100), alpha_(70.0), maxstep_(0.04), fmax_(0.05),
  use_armijo_(true), c1_(0.23), c2_(0.46), a_min_(1e-10),
  use_precon_(false), precon_r_cut_(-1.0), precon_mu_(-1.0),
  precon_A_(3.0), precon_c_stab_(0.1),
  e0_(0.0), e1_(0.0), alpha_k_(1.0), iteration_(0),
  reset_flag_(false), natoms_total_(0), ndof_(0),
  linesearch_debug_(0),
  precon_(nullptr), pe_compute_(nullptr), list(nullptr), logfile_(nullptr)
{
  // Require at least: fix ID group precon_lbfgs fmax
  if (narg < 4) error->all(FLERR,"Illegal fix precon_lbfgs command");

  // Parse fmax
  fmax_ = utils::numeric(FLERR,arg[3],false,lmp);
  if (fmax_ <= 0.0)
    error->all(FLERR,"Fix precon_lbfgs fmax must be positive");

  // Parse optional keywords
  parse_arguments(narg - 4, &arg[4]);

  H0_ = 1.0 / alpha_;

  // This fix operates on forces
  thermo_energy = 0;

  // Count degrees of freedom
  bigint count = group->count(igroup);
  natoms_total_ = static_cast<int>(count);
  ndof_ = 3 * natoms_total_;

  if (natoms_total_ == 0)
    error->all(FLERR,"Fix precon_lbfgs group has no atoms");

  // Allocate arrays
  allocate_arrays();

  // Open logfile if specified
  if (!logfilename_.empty()) {
    if (comm->me == 0) {
      logfile_ = fopen(logfilename_.c_str(), "w");
      if (!logfile_)
        error->one(FLERR,"Cannot open fix precon_lbfgs logfile");
      fprintf(logfile_, "# PreconLBFGS optimization log\n");
      fprintf(logfile_, "# Step  Energy        Fmax\n");
    }
  }
}

/* ---------------------------------------------------------------------- */

FixPreconLBFGS::~FixPreconLBFGS()
{
  if (logfile_ && comm->me == 0) fclose(logfile_);
  if (precon_) delete precon_;
}

/* ---------------------------------------------------------------------- */

int FixPreconLBFGS::setmask()
{
  int mask = 0;
  mask |= MIN_ENERGY;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixPreconLBFGS::init()
{
  // Verify atom map exists (needed for preconditioner)
  if (atom->map_style == Atom::MAP_NONE)
    error->all(FLERR,"Fix precon_lbfgs requires an atom map");

  // Get potential energy compute
  pe_compute_ = modify->get_compute_by_id("thermo_pe");
  if (!pe_compute_)
    error->all(FLERR,"Fix precon_lbfgs could not find thermo_pe compute");

  // Check neighbor list cutoff if using preconditioner
  if (use_precon_ && precon_r_cut_ > 0.0) {
    double neighbor_cutoff = neighbor->cutneighmax;
    if (neighbor_cutoff < precon_r_cut_) {
      error->all(FLERR,"LAMMPS neighbor cutoff too small for preconditioner. "
                      "Increase pair_style cutoff or decrease precon r_cut");
    }
  }

  // Initialize preconditioner (will be done lazily in min_setup)
  reset_hessian();
}

/* ---------------------------------------------------------------------- */

void FixPreconLBFGS::setup(int /*vflag*/)
{
  // Called before run
}

/* ---------------------------------------------------------------------- */

void FixPreconLBFGS::min_setup(int /*vflag*/)
{
  // Initialize for minimization
  reset_hessian();
  iteration_ = 0;

  // Get initial positions and forces
  get_positions_flat(x0_.data());
  get_forces_flat(g0_.data());
  e0_ = compute_energy();

  // Negate forces to get gradient
  for (int i = 0; i < ndof_; i++) {
    g0_[i] = -g0_[i];
  }

  if (comm->me == 0) {
    printf("PreconLBFGS: Starting optimization\n");
    printf("  Atoms: %d, DOF: %d\n", natoms_total_, ndof_);
    printf("  fmax: %.6f, maxstep: %.6f\n", fmax_, maxstep_);
    printf("  Preconditioner: %s\n", use_precon_ ? "Exp" : "Identity");
  }

  // Initialize preconditioner if requested
  if (use_precon_) {
    if (!precon_) {
      precon_ = new PreconExp(precon_r_cut_, precon_mu_, precon_A_, precon_c_stab_);
    }

    // Detect fixed atoms
    std::vector<int> fixed_tags = detect_fixed_atoms();

    if (comm->me == 0 && fixed_tags.size() > 0) {
      printf("  Fixed atoms detected: %zu\n", fixed_tags.size());
    }

    // Build preconditioner matrix
    precon_->make_precon(lmp, groupbit, fixed_tags, list);

    if (comm->me == 0) {
      printf("  Preconditioner built: r_cut=%.3f, r_NN=%.3f, mu=%.3f\n",
             precon_->r_cut, precon_->r_NN, precon_->mu);
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixPreconLBFGS::min_step(double alpha, double * /*fextra*/)
{
  // This is called by LAMMPS minimizer to try a step
  // We do our own line search in min_post_force(), so we ignore LAMMPS's alpha
  // Do nothing - keep positions as we set them in min_post_force()

  if (linesearch_debug_ && comm->me == 0) {
    printf("[MIN_STEP] Called with alpha=%.10e (ignored), iteration=%d\n", alpha, iteration_);
  }

  // Do nothing - we handle all position updates in min_post_force()
}

/* ---------------------------------------------------------------------- */

void FixPreconLBFGS::min_post_force(int /*vflag*/)
{
  // Called after forces are computed
  // Read current state, perform LBFGS update and line search

  std::vector<double> x(ndof_);
  std::vector<double> g(ndof_);

  get_positions_flat(x.data());
  get_forces_flat(g.data());

  // Negate forces to get gradient
  for (int i = 0; i < ndof_; i++) {
    g[i] = -g[i];
  }

  double e = compute_energy();

  if (linesearch_debug_ && comm->me == 0) {
    printf("[POST_FORCE] iteration=%d, E=%.10f, fmax=%.6f\n",
           iteration_, e, compute_fmax(g.data()));
  }

  // Update history if not first step
  // History uses: s = x_current - x_previous, y = g_current - g_previous
  if (!reset_flag_ && iteration_ > 0) {
    update_history(x.data(), g.data());
  }
  reset_flag_ = false;

  // Store current state for next iteration's history update
  // This must be done BEFORE line search
  x0_ = x;
  g0_ = g;
  e0_ = e;

  // Compute search direction using LBFGS two-loop
  compute_search_direction(g.data());

  // Perform line search from current position x
  if (use_armijo_) {
    alpha_k_ = armijo_line_search(x.data(), p_.data(), g.data(), e);
  } else {
    // TODO: Implement Wolfe line search
    alpha_k_ = 1.0;  // Placeholder
  }

  // Take accepted step
  std::vector<double> x_new(ndof_);
  for (int i = 0; i < ndof_; i++) {
    x_new[i] = x[i] + alpha_k_ * p_[i];
  }

  set_positions_flat(x_new.data());

  iteration_++;

  // Log progress (using gradient before step)
  double fmax = compute_fmax(g.data());
  write_log(e, fmax);
}

/* ---------------------------------------------------------------------- */

double FixPreconLBFGS::min_energy(double *fextra)
{
  if (fextra) *fextra = 0.0;
  return compute_energy();
}

/* ---------------------------------------------------------------------- */

void FixPreconLBFGS::min_store()
{
  // Store current state
}

/* ---------------------------------------------------------------------- */

void FixPreconLBFGS::min_clearstore()
{
  // Clear stored state
}

/* ---------------------------------------------------------------------- */

void FixPreconLBFGS::min_pushstore()
{
  // Push to history
}

/* ---------------------------------------------------------------------- */

int FixPreconLBFGS::min_reset_ref()
{
  // Reset reference state
  return 0;
}

/* ---------------------------------------------------------------------- */

double FixPreconLBFGS::max_alpha(double * /*dx*/)
{
  // Maximum alpha based on maxstep
  double max_disp = 0.0;
  for (int i = 0; i < ndof_; i++) {
    max_disp = std::max(max_disp, std::abs(p_[i]));
  }

  if (max_disp > 1e-10) {
    return (maxstep_ * std::sqrt(static_cast<double>(natoms_total_))) / max_disp;
  }
  return 1.0;
}

/* ---------------------------------------------------------------------- */

int FixPreconLBFGS::min_dof()
{
  return ndof_;
}

/* ----------------------------------------------------------------------
   Private methods
---------------------------------------------------------------------- */

void FixPreconLBFGS::parse_arguments(int narg, char **arg)
{
  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"memory") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix precon_lbfgs command");
      memory_ = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      if (memory_ < 1) error->all(FLERR,"Fix precon_lbfgs memory must be >= 1");
      iarg += 2;
    } else if (strcmp(arg[iarg],"alpha") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix precon_lbfgs command");
      alpha_ = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      if (alpha_ <= 0.0) error->all(FLERR,"Fix precon_lbfgs alpha must be positive");
      iarg += 2;
    } else if (strcmp(arg[iarg],"maxstep") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix precon_lbfgs command");
      maxstep_ = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      if (maxstep_ <= 0.0) error->all(FLERR,"Fix precon_lbfgs maxstep must be positive");
      iarg += 2;
    } else if (strcmp(arg[iarg],"precon") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix precon_lbfgs command");
      if (strcmp(arg[iarg+1],"exp") == 0) {
        use_precon_ = true;
      } else if (strcmp(arg[iarg+1],"none") == 0) {
        use_precon_ = false;
      } else {
        error->all(FLERR,"Fix precon_lbfgs precon must be 'exp' or 'none'");
      }
      iarg += 2;
    } else if (strcmp(arg[iarg],"r_cut") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix precon_lbfgs command");
      precon_r_cut_ = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"mu") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix precon_lbfgs command");
      precon_mu_ = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"A") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix precon_lbfgs command");
      precon_A_ = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"c_stab") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix precon_lbfgs command");
      precon_c_stab_ = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"c1") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix precon_lbfgs command");
      c1_ = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"linesearch_debug") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix precon_lbfgs command");
      linesearch_debug_ = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"logfile") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix precon_lbfgs command");
      logfilename_ = arg[iarg+1];
      iarg += 2;
    } else {
      error->all(FLERR,"Illegal fix precon_lbfgs command");
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixPreconLBFGS::allocate_arrays()
{
  x0_.resize(ndof_);
  g0_.resize(ndof_);
  p_.resize(ndof_);
}

/* ---------------------------------------------------------------------- */

void FixPreconLBFGS::reset_hessian()
{
  s_.clear();
  y_.clear();
  rho_.clear();
  reset_flag_ = true;
}

/* ---------------------------------------------------------------------- */

void FixPreconLBFGS::compute_search_direction(const double *gradient)
{
  lbfgs_two_loop(gradient, p_.data());

  // Negate to get descent direction
  for (int i = 0; i < ndof_; i++) {
    p_[i] = -p_[i];
  }
}

/* ---------------------------------------------------------------------- */

void FixPreconLBFGS::lbfgs_two_loop(const double *gradient, double *direction)
{
  int loopmax = std::min(memory_, static_cast<int>(y_.size()));
  std::vector<double> a(loopmax);
  std::vector<double> q(ndof_);

  // q = -gradient (we'll negate direction at the end)
  for (int i = 0; i < ndof_; i++) {
    q[i] = gradient[i];
  }

  // First loop (backward)
  for (int i = loopmax - 1; i >= 0; i--) {
    a[i] = rho_[i] * dot_product(s_[i], q);
    for (int j = 0; j < ndof_; j++) {
      q[j] -= a[i] * y_[i][j];
    }
  }

  // Apply preconditioner or H0
  if (use_precon_ && precon_) {
    // Use exponential preconditioner: solve P * direction = q
    precon_->solve(q.data(), direction, ndof_);
  } else {
    // Identity preconditioner: direction = H0 * q
    for (int i = 0; i < ndof_; i++) {
      direction[i] = H0_ * q[i];
    }
  }

  // Second loop (forward)
  for (int i = 0; i < loopmax; i++) {
    double b = rho_[i] * dot_product(y_[i], std::vector<double>(direction, direction + ndof_));
    for (int j = 0; j < ndof_; j++) {
      direction[j] += s_[i][j] * (a[i] - b);
    }
  }
}

/* ---------------------------------------------------------------------- */

double FixPreconLBFGS::armijo_line_search(const double *x, const double *p,
                                         const double *g, double e_current)
{
  // Compute initial slope
  double slope = 0.0;
  for (int i = 0; i < ndof_; i++) {
    slope += g[i] * p[i];
  }

  if (linesearch_debug_ && comm->me == 0) {
    printf("\n[LS DEBUG] Step %d Line Search Start\n", iteration_);
    printf("[LS DEBUG]   E_current = %.10f\n", e_current);
    printf("[LS DEBUG]   slope = %.10e (should be negative)\n", slope);
  }

  if (slope >= 0.0) {
    // Not a descent direction - reset
    if (comm->me == 0) {
      printf("WARNING: Not a descent direction (slope=%.6e), resetting Hessian\n", slope);
    }
    reset_hessian();
    return 0.0;
  }

  // Scale step to respect maxstep
  double alpha_max = max_alpha(nullptr);
  double alpha = std::min(1.0, alpha_max);

  if (linesearch_debug_ && comm->me == 0) {
    printf("[LS DEBUG]   alpha_max = %.10f\n", alpha_max);
    printf("[LS DEBUG]   alpha_init = %.10f\n", alpha);
    printf("[LS DEBUG]   c1 = %.6f, a_min = %.6e\n", c1_, a_min_);
  }

  const double rho = 0.5;  // Backtrack factor
  const int max_iter = 20;

  std::vector<double> x_trial(ndof_);

  for (int iter = 0; iter < max_iter; iter++) {
    // Try step
    for (int i = 0; i < ndof_; i++) {
      x_trial[i] = x[i] + alpha * p[i];
    }

    set_positions_flat(x_trial.data());
    double e_trial = compute_trial_energy();

    // Compute Armijo threshold
    double armijo_threshold = e_current + c1_ * alpha * slope;
    double energy_decrease = e_current - e_trial;
    double required_decrease = -c1_ * alpha * slope;

    if (linesearch_debug_ && comm->me == 0) {
      printf("[LS DEBUG]   iter %2d: alpha = %.10e\n", iter, alpha);
      printf("[LS DEBUG]           E_trial = %.10f\n", e_trial);
      printf("[LS DEBUG]           E_current - E_trial = %.10e\n", energy_decrease);
      printf("[LS DEBUG]           Required decrease = %.10e\n", required_decrease);
      printf("[LS DEBUG]           Armijo threshold = %.10f\n", armijo_threshold);
      printf("[LS DEBUG]           Condition: %.10f <= %.10f ? %s\n",
             e_trial, armijo_threshold,
             (e_trial <= armijo_threshold) ? "YES" : "NO");
    }

    // Check Armijo condition
    if (e_trial <= armijo_threshold) {
      if (linesearch_debug_ && comm->me == 0) {
        printf("[LS DEBUG]   *** ACCEPTED at iter %d with alpha = %.10e ***\n", iter, alpha);
      }
      e1_ = e_trial;
      // Keep trial positions set
      return alpha;  // Accept
    }

    // Reject this trial - restore original positions before trying smaller alpha
    set_positions_flat(x);

    // Reduce step for next iteration
    alpha *= rho;

    if (alpha < a_min_) {
      if (comm->me == 0) {
        printf("WARNING: Line search failed (alpha = %.6e < a_min = %.6e)\n", alpha, a_min_);
        printf("         Taking small step with a_min\n");
        if (linesearch_debug_) {
          printf("[LS DEBUG]   Last E_trial = %.10f, E_current = %.10f\n", e_trial, e_current);
          printf("[LS DEBUG]   Last energy decrease = %.10e\n", energy_decrease);
          printf("[LS DEBUG]   Last required decrease = %.10e\n", required_decrease);
        }
      }
      e1_ = e_trial;
      return a_min_;
    }
  }

  // Max iterations reached, take what we have
  if (linesearch_debug_ && comm->me == 0) {
    printf("[LS DEBUG]   *** MAX ITERATIONS reached, using alpha = %.10e ***\n", alpha);
  }
  e1_ = compute_trial_energy();
  return alpha;
}

/* ---------------------------------------------------------------------- */

void FixPreconLBFGS::update_history(const double *x, const double *g)
{
  // s = x - x0
  std::vector<double> s(ndof_);
  for (int i = 0; i < ndof_; i++) {
    s[i] = x[i] - x0_[i];
  }

  // y = g - g0
  std::vector<double> y(ndof_);
  for (int i = 0; i < ndof_; i++) {
    y[i] = g[i] - g0_[i];
  }

  // rho = 1 / (y · s)
  double ys_dot = dot_product(y, s);
  if (std::abs(ys_dot) < 1e-10) {
    // Skip this update
    return;
  }
  double rho = 1.0 / ys_dot;

  // Add to history
  s_.push_back(s);
  y_.push_back(y);
  rho_.push_back(rho);

  // Remove oldest if exceeds memory
  if (static_cast<int>(s_.size()) > memory_) {
    s_.pop_front();
    y_.pop_front();
    rho_.pop_front();
  }
}

/* ---------------------------------------------------------------------- */

void FixPreconLBFGS::get_positions_flat(double *x) const
{
  double **atom_x = atom->x;
  int *mask = atom->mask;
  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;

  // Count local atoms in group
  int nlocal_group = 0;
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) nlocal_group++;
  }

  // Gather local positions and tags
  std::vector<double> x_local(3 * nlocal_group);
  std::vector<tagint> tags_local(nlocal_group);

  int idx = 0;
  int atom_idx = 0;
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      tags_local[atom_idx] = tag[i];
      x_local[idx++] = atom_x[i][0];
      x_local[idx++] = atom_x[i][1];
      x_local[idx++] = atom_x[i][2];
      atom_idx++;
    }
  }

  // Gather counts from all ranks
  int nprocs = comm->nprocs;
  std::vector<int> recvcounts(nprocs);
  MPI_Allgather(&nlocal_group, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, world);

  // Compute displacements for positions (3 * atom counts)
  std::vector<int> displs(nprocs);
  std::vector<int> recvcounts_pos(nprocs);
  displs[0] = 0;
  recvcounts_pos[0] = 3 * recvcounts[0];
  for (int i = 1; i < nprocs; i++) {
    displs[i] = displs[i-1] + 3 * recvcounts[i-1];
    recvcounts_pos[i] = 3 * recvcounts[i];
  }

  // Gather all positions
  std::vector<double> x_all(3 * natoms_total_);
  MPI_Allgatherv(x_local.data(), 3 * nlocal_group, MPI_DOUBLE,
                 x_all.data(), recvcounts_pos.data(), displs.data(),
                 MPI_DOUBLE, world);

  // Gather all tags
  std::vector<int> displs_tags(nprocs);
  displs_tags[0] = 0;
  for (int i = 1; i < nprocs; i++) {
    displs_tags[i] = displs_tags[i-1] + recvcounts[i-1];
  }

  std::vector<tagint> tags_all(natoms_total_);
  MPI_Allgatherv(tags_local.data(), nlocal_group, MPI_LMP_TAGINT,
                 tags_all.data(), recvcounts.data(), displs_tags.data(),
                 MPI_LMP_TAGINT, world);

  // Sort by tag and reorder positions
  std::vector<std::pair<tagint, int>> tag_idx(natoms_total_);
  for (int i = 0; i < natoms_total_; i++) {
    tag_idx[i] = {tags_all[i], i};
  }
  std::sort(tag_idx.begin(), tag_idx.end());

  // Reorder into output array
  for (int i = 0; i < natoms_total_; i++) {
    int src_idx = tag_idx[i].second;
    x[3*i + 0] = x_all[3*src_idx + 0];
    x[3*i + 1] = x_all[3*src_idx + 1];
    x[3*i + 2] = x_all[3*src_idx + 2];
  }
}

/* ---------------------------------------------------------------------- */

void FixPreconLBFGS::set_positions_flat(const double *x)
{
  double **atom_x = atom->x;
  int *mask = atom->mask;
  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;

  // Debug: Print position before update
  if (linesearch_debug_ && comm->me == 0 && nlocal > 0) {
    // Find first atom in group
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        printf("[POS DEBUG] Before update: atom tag=%d pos=(%.10f, %.10f, %.10f)\n",
               tag[i], atom_x[i][0], atom_x[i][1], atom_x[i][2]);
        break;
      }
    }
  }

  // Build tag-to-index map for this rank's atoms
  // Assumes input x is sorted by tag (tag 1, 2, 3, ...)
  std::map<tagint, int> tag_to_local;
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      tag_to_local[tag[i]] = i;
    }
  }

  // Gather all tags from all ranks to determine global ordering
  int nlocal_group = tag_to_local.size();
  int nprocs = comm->nprocs;

  std::vector<int> recvcounts(nprocs);
  MPI_Allgather(&nlocal_group, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, world);

  std::vector<int> displs_tags(nprocs);
  displs_tags[0] = 0;
  for (int i = 1; i < nprocs; i++) {
    displs_tags[i] = displs_tags[i-1] + recvcounts[i-1];
  }

  std::vector<tagint> tags_local(nlocal_group);
  int idx = 0;
  for (const auto &pair : tag_to_local) {
    tags_local[idx++] = pair.first;
  }

  std::vector<tagint> tags_all(natoms_total_);
  MPI_Allgatherv(tags_local.data(), nlocal_group, MPI_LMP_TAGINT,
                 tags_all.data(), recvcounts.data(), displs_tags.data(),
                 MPI_LMP_TAGINT, world);

  // Sort tags to get global ordering
  std::vector<std::pair<tagint, int>> tag_idx(natoms_total_);
  for (int i = 0; i < natoms_total_; i++) {
    tag_idx[i] = {tags_all[i], i};
  }
  std::sort(tag_idx.begin(), tag_idx.end());

  // Build reverse map: tag -> position in sorted array
  std::map<tagint, int> tag_to_sorted_idx;
  for (int i = 0; i < natoms_total_; i++) {
    tag_to_sorted_idx[tag_idx[i].first] = i;
  }

  // Set positions for local atoms
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      int sorted_idx = tag_to_sorted_idx[tag[i]];
      atom_x[i][0] = x[3*sorted_idx + 0];
      atom_x[i][1] = x[3*sorted_idx + 1];
      atom_x[i][2] = x[3*sorted_idx + 2];
    }
  }

  // Debug: Print position after update
  if (linesearch_debug_ && comm->me == 0 && nlocal > 0) {
    // Find first atom in group
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        printf("[POS DEBUG] After update: atom tag=%d pos=(%.10f, %.10f, %.10f)\n",
               tag[i], atom_x[i][0], atom_x[i][1], atom_x[i][2]);
        break;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixPreconLBFGS::get_forces_flat(double *f) const
{
  double **atom_f = atom->f;
  int *mask = atom->mask;
  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;

  // Count local atoms in group
  int nlocal_group = 0;
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) nlocal_group++;
  }

  // Gather local forces and tags
  std::vector<double> f_local(3 * nlocal_group);
  std::vector<tagint> tags_local(nlocal_group);

  int idx = 0;
  int atom_idx = 0;
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      tags_local[atom_idx] = tag[i];
      f_local[idx++] = atom_f[i][0];
      f_local[idx++] = atom_f[i][1];
      f_local[idx++] = atom_f[i][2];
      atom_idx++;
    }
  }

  // Gather counts from all ranks
  int nprocs = comm->nprocs;
  std::vector<int> recvcounts(nprocs);
  MPI_Allgather(&nlocal_group, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, world);

  // Compute displacements for forces (3 * atom counts)
  std::vector<int> displs(nprocs);
  std::vector<int> recvcounts_forces(nprocs);
  displs[0] = 0;
  recvcounts_forces[0] = 3 * recvcounts[0];
  for (int i = 1; i < nprocs; i++) {
    displs[i] = displs[i-1] + 3 * recvcounts[i-1];
    recvcounts_forces[i] = 3 * recvcounts[i];
  }

  // Gather all forces
  std::vector<double> f_all(3 * natoms_total_);
  MPI_Allgatherv(f_local.data(), 3 * nlocal_group, MPI_DOUBLE,
                 f_all.data(), recvcounts_forces.data(), displs.data(),
                 MPI_DOUBLE, world);

  // Gather all tags
  std::vector<int> displs_tags(nprocs);
  displs_tags[0] = 0;
  for (int i = 1; i < nprocs; i++) {
    displs_tags[i] = displs_tags[i-1] + recvcounts[i-1];
  }

  std::vector<tagint> tags_all(natoms_total_);
  MPI_Allgatherv(tags_local.data(), nlocal_group, MPI_LMP_TAGINT,
                 tags_all.data(), recvcounts.data(), displs_tags.data(),
                 MPI_LMP_TAGINT, world);

  // Sort by tag and reorder forces
  std::vector<std::pair<tagint, int>> tag_idx(natoms_total_);
  for (int i = 0; i < natoms_total_; i++) {
    tag_idx[i] = {tags_all[i], i};
  }
  std::sort(tag_idx.begin(), tag_idx.end());

  // Reorder into output array
  for (int i = 0; i < natoms_total_; i++) {
    int src_idx = tag_idx[i].second;
    f[3*i + 0] = f_all[3*src_idx + 0];
    f[3*i + 1] = f_all[3*src_idx + 1];
    f[3*i + 2] = f_all[3*src_idx + 2];
  }
}

/* ---------------------------------------------------------------------- */

double FixPreconLBFGS::compute_energy()
{
  // Read current energy (assumes forces already computed by LAMMPS)
  return pe_compute_->compute_scalar();
}

/* ---------------------------------------------------------------------- */

double FixPreconLBFGS::compute_trial_energy()
{
  // For trial positions during line search, we must trigger force recomputation
  // Otherwise energy will be stale/cached value

  // Clear forces
  double **f = atom->f;
  int nall = atom->nlocal + atom->nghost;
  for (int i = 0; i < nall; i++) {
    f[i][0] = 0.0;
    f[i][1] = 0.0;
    f[i][2] = 0.0;
  }

  // Trigger force computation
  lmp->force->pair->compute(lmp->update->ntimestep, 1);

  // Reverse communicate forces (needed for MPI)
  comm->reverse_comm();

  // Now read energy (will be recomputed)
  double energy = pe_compute_->compute_scalar();

  if (linesearch_debug_ && comm->me == 0) {
    printf("[ENERGY DEBUG] Trial energy = %.10f\n", energy);
  }

  return energy;
}

/* ---------------------------------------------------------------------- */

double FixPreconLBFGS::compute_fmax(const double *forces) const
{
  // Compute max absolute force component
  double fmax_local = 0.0;
  for (int i = 0; i < ndof_; i++) {
    fmax_local = std::max(fmax_local, std::abs(forces[i]));
  }

  // MPI reduction
  double fmax_global = fmax_local;
  MPI_Allreduce(&fmax_local, &fmax_global, 1, MPI_DOUBLE, MPI_MAX, world);

  return fmax_global;
}

/* ---------------------------------------------------------------------- */

void FixPreconLBFGS::write_log(double energy, double fmax)
{
  if (logfile_ && comm->me == 0) {
    fprintf(logfile_, "%6d  %16.8f  %12.6f\n", iteration_, energy, fmax);
    fflush(logfile_);
  }

  if (comm->me == 0 && iteration_ % 10 == 0) {
    printf("PreconLBFGS: step %d, E = %.6f, fmax = %.6f\n",
           iteration_, energy, fmax);
  }
}

/* ---------------------------------------------------------------------- */

std::vector<int> FixPreconLBFGS::detect_fixed_atoms() const
{
  std::vector<int> fixed_tags;

  // Check for FixSetForce - atoms with all components set to 0
  for (int i = 0; i < modify->nfix; i++) {
    Fix *fix = modify->fix[i];

    // Check if it's a setforce fix
    if (strcmp(fix->style, "setforce") == 0) {
      // Iterate through atoms in our group
      int *mask = atom->mask;
      tagint *tag = atom->tag;
      int nlocal = atom->nlocal;

      for (int j = 0; j < nlocal; j++) {
        if (mask[j] & groupbit) {
          // Check if this atom is affected by the setforce fix
          // If setforce is set to NULL NULL NULL or 0 0 0, atom is fixed
          // Note: This is a simplified check - full implementation would
          // need to inspect the fix's internal state

          // For now, mark all atoms in setforce group as potentially fixed
          if (mask[j] & fix->groupbit) {
            fixed_tags.push_back(tag[j]);
          }
        }
      }
    }

    // Check for freeze fix
    if (strcmp(fix->style, "freeze") == 0) {
      int *mask = atom->mask;
      tagint *tag = atom->tag;
      int nlocal = atom->nlocal;

      for (int j = 0; j < nlocal; j++) {
        if ((mask[j] & groupbit) && (mask[j] & fix->groupbit)) {
          fixed_tags.push_back(tag[j]);
        }
      }
    }
  }

  // Gather fixed tags from all ranks
  int nlocal_fixed = fixed_tags.size();
  int nprocs = comm->nprocs;

  std::vector<int> recvcounts(nprocs);
  MPI_Allgather(&nlocal_fixed, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, world);

  int total_fixed = 0;
  std::vector<int> displs(nprocs);
  for (int i = 0; i < nprocs; i++) {
    displs[i] = total_fixed;
    total_fixed += recvcounts[i];
  }

  std::vector<int> all_fixed_tags(total_fixed);
  MPI_Allgatherv(fixed_tags.data(), nlocal_fixed, MPI_INT,
                 all_fixed_tags.data(), recvcounts.data(), displs.data(),
                 MPI_INT, world);

  // Remove duplicates
  std::set<int> unique_fixed(all_fixed_tags.begin(), all_fixed_tags.end());
  std::vector<int> result(unique_fixed.begin(), unique_fixed.end());

  return result;
}

/* ---------------------------------------------------------------------- */

double FixPreconLBFGS::dot_product(const std::vector<double> &a,
                                  const std::vector<double> &b) const
{
  double result = 0.0;
  for (size_t i = 0; i < a.size(); i++) {
    result += a[i] * b[i];
  }
  return result;
}

/* ---------------------------------------------------------------------- */

double FixPreconLBFGS::vector_norm(const std::vector<double> &v) const
{
  return std::sqrt(dot_product(v, v));
}

/* ---------------------------------------------------------------------- */

double FixPreconLBFGS::vector_max_abs(const std::vector<double> &v) const
{
  double max_val = 0.0;
  for (const auto &val : v) {
    max_val = std::max(max_val, std::abs(val));
  }
  return max_val;
}

/* ---------------------------------------------------------------------- */

void FixPreconLBFGS::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}
