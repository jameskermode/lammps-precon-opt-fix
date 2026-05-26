/* Stage 8b/8c — `min_style precon/lbfgs`, MPI-aware + variable-cell.
   See min_precon_lbfgs.h. */
#include "min_precon_lbfgs.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "fix_minimize.h"
#include "fix_precon_exp.h"
#include "modify.h"
#include "neighbor.h"
#include "output.h"
#include "thermo.h"
#include "timer.h"
#include "update.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

static constexpr double EPS_ENERGY = 1.0e-8;
static constexpr double EPS_YS = 1.0e-10;  // LBFGS curvature-condition floor

static constexpr double EXP_A = 3.0;
static constexpr double EXP_C_STAB = 0.1;

// linemin_armijo constants — chosen to mirror ASE LineSearchArmijo behaviour.
static constexpr double ARMIJO_C1 = 0.1;        // Armijo slope (vs LAMMPS 0.4)
static constexpr double ARMIJO_ALPHA_MAX = 1.0; // clamp on initial alpha
static constexpr double ARMIJO_REDUCE_MIN = 0.1; // floor on backtrack ratio
static constexpr double ARMIJO_REDUCE_MAX = 0.5; // ceiling on backtrack ratio
static constexpr double ARMIJO_EMACH = 1.0e-8;

/* ---------------------------------------------------------------------- */

MinPreconLBFGS::MinPreconLBFGS(LAMMPS *lmp) : MinLineSearch(lmp) {
  // Default the per-atom step cap to 1.0 A (LAMMPS's stock Min default is 0.1).
  // With LAMMPS's Armijo-based linemins (backtrack / quadratic — the defaults
  // for min_modify line) the sufficient-decrease test alone keeps the step
  // safe, and scripts/maxstep_study.py shows the tighter caps just throttle
  // convergence on the Packwood set. Override with `min_modify dmax ...`.
  dmax = 1.0;
}

void MinPreconLBFGS::init() {
  MinLineSearch::init();
  // MinLineSearch::init() above derives `linemin` from `linestyle`. Override
  // with our looser Armijo unless the user opted out via `min_modify
  // precon_armijo off`. The cast is the legal Derived::* -> Base::* via
  // static_cast: safe because `linemin` is only invoked on a MinPreconLBFGS.
  if (use_armijo_) {
    linemin = static_cast<int (MinLineSearch::*)(double, double &)>(
        &MinPreconLBFGS::linemin_armijo);
  }
  precon_ready_ = false;
  has_prev_ = false;
  last_ncalls_ = -1;
  s_hist_.clear();
  y_hist_.clear();
  rho_hist_.clear();
  sc_hist_.clear();
  yc_hist_.clear();
  rhoc_hist_.clear();
}

/* ----------------------------------------------------------------------
   min_modify extension: `precon_armijo on|off`
   The base class's modify_params() calls this for unknown keywords. Returns
   the number of args consumed (the base then advances by that count).
------------------------------------------------------------------------- */

int MinPreconLBFGS::modify_param(int narg, char **arg) {
  if (narg >= 2 && strcmp(arg[0], "precon_armijo") == 0) {
    use_armijo_ = utils::logical(FLERR, arg[1], false, lmp);
    // The linemin pointer is reassigned on the next init(); no need to touch
    // it here, since init() runs at the start of every `minimize` command.
    return 2;
  }
  return 0;
}

/* ----------------------------------------------------------------------
   linemin_armijo: Armijo sufficient-decrease + quadratic-interpolation
   backtrack. Drop-in replacement for `linemin_backtrack` with two changes:
   (a) c1 = 0.1 (vs LAMMPS's BACKTRACK_SLOPE = 0.4); the looser test accepts
       the trial step ~90% of the time on Packwood iceVIII (vs ~50% for
       backtrack). On the other Packwood structures the preconditioned step
       is closer to Newton, so this difference is small.
   (b) On rejection, backtrack by *quadratic interpolation* of the energy
       (Nocedal & Wright Eq. 3.58) rather than constant halving — typically
       lands the next trial much closer to the Armijo-feasible region.
   Bookkeeping (fdothall/hmaxall, nextra_atom, nextra_global, box store) is
   identical to MinLineSearch::linemin_backtrack, so the function composes
   correctly with `fix box/relax` (Stage 8c).
------------------------------------------------------------------------- */

int MinPreconLBFGS::linemin_armijo(double eoriginal, double &alpha) {
  int i, m, n;
  double fdothall, fdothme, hme, hmax, hmaxall;
  double *xatom, *x0atom, *fatom, *hatom;

  // fdothall = projection of search dir along downhill gradient
  // (positive => h is a descent direction)
  fdothme = 0.0;
  for (i = 0; i < nvec; i++) fdothme += fvec[i] * h[i];
  if (nextra_atom)
    for (m = 0; m < nextra_atom; m++) {
      fatom = fextra_atom[m];
      hatom = hextra_atom[m];
      n = extra_nlen[m];
      for (i = 0; i < n; i++) fdothme += fatom[i] * hatom[i];
    }
  MPI_Allreduce(&fdothme, &fdothall, 1, MPI_DOUBLE, MPI_SUM, world);
  if (nextra_global)
    for (i = 0; i < nextra_global; i++) fdothall += fextra[i] * hextra[i];
  if (output->thermo->normflag) fdothall /= atom->natoms;
  if (fdothall <= 0.0) return DOWNHILL;

  // Initial alpha clamped by dmax (atomic) + extra_max[] (per-atom) + fix
  // (global). Identical to MinLineSearch::linemin_backtrack.
  hme = 0.0;
  for (i = 0; i < nvec; i++) hme = std::max(hme, std::fabs(h[i]));
  MPI_Allreduce(&hme, &hmaxall, 1, MPI_DOUBLE, MPI_MAX, world);
  alpha = std::min(ARMIJO_ALPHA_MAX, dmax / hmaxall);
  if (nextra_atom)
    for (m = 0; m < nextra_atom; m++) {
      hatom = hextra_atom[m];
      n = extra_nlen[m];
      hme = 0.0;
      for (i = 0; i < n; i++) hme = std::max(hme, std::fabs(hatom[i]));
      MPI_Allreduce(&hme, &hmax, 1, MPI_DOUBLE, MPI_MAX, world);
      alpha = std::min(alpha, extra_max[m] / hmax);
      hmaxall = std::max(hmaxall, hmax);
    }
  if (nextra_global) {
    double alpha_extra = modify->max_alpha(hextra);
    alpha = std::min(alpha, alpha_extra);
    for (i = 0; i < nextra_global; i++)
      hmaxall = std::max(hmaxall, std::fabs(hextra[i]));
  }
  if (hmaxall == 0.0) return ZEROFORCE;

  // Store box and current positions so alpha_step can build x = x0 + alpha*h.
  fix_minimize->store_box();
  for (i = 0; i < nvec; i++) x0[i] = xvec[i];
  if (nextra_atom)
    for (m = 0; m < nextra_atom; m++) {
      xatom = xextra_atom[m];
      x0atom = x0extra_atom[m];
      n = extra_nlen[m];
      for (i = 0; i < n; i++) x0atom[i] = xatom[i];
    }
  if (nextra_global) modify->min_store();

  // Armijo loop with quadratic-interpolation backtrack.
  // In textbook notation g0 = dE/dalpha|_0 = -fdothall (< 0 for descent).
  const double g0 = -fdothall;
  while (true) {
    ecurrent = alpha_step(alpha, 1);

    // Armijo: ecurrent <= eoriginal + c1 * alpha * g0
    //       = eoriginal - c1 * alpha * fdothall
    const double de = ecurrent - eoriginal;
    const double de_ideal = -ARMIJO_C1 * alpha * fdothall;
    if (de <= de_ideal) {
      if (nextra_global) {
        int itmp = modify->min_reset_ref();
        if (itmp) ecurrent = energy_force(1);
      }
      return 0;
    }

    // Quadratic interpolation: fit parabola through (0, eoriginal, g0) and
    // (alpha, ecurrent); its minimizer is at
    //     alpha_q = -g0 * alpha^2 / (2 * (ecurrent - eoriginal - g0*alpha))
    // The denominator is the parabola's curvature (positive when the model
    // has a minimum). Clamp the resulting backtrack ratio to a safe band so
    // we neither under-shrink (ratio too close to 1 -> infinite loop) nor
    // collapse (ratio too small -> wasted trials).
    const double denom = 2.0 * (ecurrent - eoriginal - g0 * alpha);
    double alpha_new;
    if (denom > 0.0) {
      alpha_new = -g0 * alpha * alpha / denom;
    } else {
      // Non-convex local model -> halve and try again.
      alpha_new = ARMIJO_REDUCE_MAX * alpha;
    }
    alpha_new = std::min(std::max(alpha_new, ARMIJO_REDUCE_MIN * alpha),
                         ARMIJO_REDUCE_MAX * alpha);
    alpha = alpha_new;

    // Backtracked too far -> give up. Mirror MinLineSearch::linemin_backtrack.
    if (alpha <= 0.0 || de_ideal >= -ARMIJO_EMACH) {
      ecurrent = alpha_step(0.0, 0);
      if (de < 0.0) return ETOL;
      return ZEROALPHA;
    }
  }
}

double MinPreconLBFGS::ddot(const double *a, const double *b, int n) const {
  double local = 0.0;
  for (int k = 0; k < n; ++k) local += a[k] * b[k];
  double global = 0.0;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, world);
  return global;
}

/* ----------------------------------------------------------------------
   distributed two-force-call finite-difference estimate of mu (positions);
   plus a finite-difference probe of the cell-DOF metric mu_c
------------------------------------------------------------------------- */

void MinPreconLBFGS::estimate_mu() {
  const int n = nvec;

  double pmin[3] = {1e300, 1e300, 1e300};
  double pmax[3] = {-1e300, -1e300, -1e300};
  for (int k = 0; k < n; ++k) {
    const int d = k % 3;
    if (xvec[k] < pmin[d]) pmin[d] = xvec[k];
    if (xvec[k] > pmax[d]) pmax[d] = xvec[k];
  }
  double gmin[3], gmax[3];
  MPI_Allreduce(pmin, gmin, 3, MPI_DOUBLE, MPI_MIN, world);
  MPI_Allreduce(pmax, gmax, 3, MPI_DOUBLE, MPI_MAX, world);

  const double H = 1.0e-2 * fix_->r_NN();
  std::vector<double> v(n);
  for (int k = 0; k < n; ++k) {
    const int d = k % 3;
    const double L = gmax[d] - gmin[d];
    v[k] = (L > 0.0) ? H * std::sin(xvec[k] / L) : 0.0;
  }

  // forces at p (current) and at p + v; the cell force fextra is preserved
  std::vector<double> f_p(fvec, fvec + n);
  std::vector<double> fe_p(fextra, fextra + nextra_global);
  const double e_p = ecurrent;
  for (int k = 0; k < n; ++k) xvec[k] += v[k];
  energy_force(1);
  double lhs_local = 0.0;
  for (int k = 0; k < n; ++k) lhs_local += (f_p[k] - fvec[k]) * v[k];  // dE=-f
  for (int k = 0; k < n; ++k) xvec[k] -= v[k];   // restore geometry...
  for (int k = 0; k < n; ++k) fvec[k] = f_p[k];  // ...atomic forces,
  for (int e = 0; e < nextra_global; ++e) fextra[e] = fe_p[e];  // cell force,
  ecurrent = e_p;                                               // and energy
  double lhs;
  MPI_Allreduce(&lhs_local, &lhs, 1, MPI_DOUBLE, MPI_SUM, world);

  std::vector<double> p1v(n);
  fix_->matvec(v.data(), p1v.data());
  const double rhs = ddot(p1v.data(), v.data(), n);

  double mu = (rhs != 0.0) ? lhs / rhs : 1.0;
  if (mu < 1.0) mu = 1.0;
  fix_->scale(mu);

  // --- cell-metric mu_c: finite-difference probe of the cell DOF ----------
  // The atomic block is preconditioned by P_pos, so the cell block needs a
  // comparable scale (mu_c ~ the cell-DOF curvature) for the line search.
  if (nextra_global > 0) {
    std::vector<double> fe0(fextra, fextra + nextra_global);
    std::vector<double> fv0(fvec, fvec + n);
    const double e0 = ecurrent;
    std::vector<double> vcell(nextra_global, 1.0);
    const double dstep = 1.0e-3;
    modify->min_store();
    modify->min_step(dstep, vcell.data());  // perturb the cell DOF
    energy_force(1);
    double lhs_c = 0.0, rhs_c = 0.0;
    for (int e = 0; e < nextra_global; ++e) {
      lhs_c += (fe0[e] - fextra[e]) * vcell[e];  // dE = -f
      rhs_c += vcell[e] * vcell[e];
    }
    modify->min_step(0.0, vcell.data());    // restore the cell...
    for (int k = 0; k < n; ++k) fvec[k] = fv0[k];                  // ...atomic
    for (int e = 0; e < nextra_global; ++e) fextra[e] = fe0[e];    // ...cell
    ecurrent = e0;                                                 // ...energy
    double mu_c = (rhs_c != 0.0) ? lhs_c / (dstep * rhs_c) : 1.0;
    if (mu_c < 1.0) mu_c = 1.0;
    fix_->set_mu_c(mu_c);
  } else {
    fix_->set_mu_c(1.0);
  }
}

/* ---------------------------------------------------------------------- */

void MinPreconLBFGS::setup_precon() {
  fix_ = nullptr;
  for (int i = 0; i < modify->nfix; ++i)
    if (strcmp(modify->fix[i]->style, "precon/exp") == 0)
      fix_ = dynamic_cast<FixPreconExp *>(modify->fix[i]);
  if (!fix_)
    error->all(FLERR, "min_style precon/lbfgs requires a 'fix precon/exp'");
  if (atom->map_style == Atom::MAP_NONE)
    error->all(FLERR,
               "min_style precon/lbfgs requires 'atom_modify map array'");

  grad_.assign(nvec, 0.0);
  g_prev_.assign(nvec, 0.0);
  x_prev_.assign(nvec, 0.0);
  gradc_.assign(nextra_global, 0.0);
  gc_prev_.assign(nextra_global, 0.0);
  s_hist_.clear();
  y_hist_.clear();
  rho_hist_.clear();
  sc_hist_.clear();
  yc_hist_.clear();
  rhoc_hist_.clear();
  has_prev_ = false;

  fix_->set_params(EXP_A, EXP_C_STAB);
  const double r_NN = fix_->compute_r_NN();
  const double r_cut =
      (fix_->user_r_cut() > 0.0) ? fix_->user_r_cut() : 2.0 * r_NN;
  fix_->set_geometry(r_NN, r_cut);
  fix_->assemble(1.0);  // P1 (mu = 1)
  estimate_mu();        // P1 -> mu * P1, and probe mu_c

  if (!fix_->dump_prefix().empty()) fix_->dump(fix_->dump_prefix());
  last_ncalls_ = neighbor->ncalls;
  precon_ready_ = true;
}

/* ----------------------------------------------------------------------
   preconditioned LBFGS iteration

   The atomic DOF and the extra global cell DOF are relaxed by *separate*
   LBFGS recursions — the preconditioner is block-diagonal and the two
   blocks have very different scales, so a combined two-loop gives
   ill-conditioned y.s dot products.  The shared line search couples them.
------------------------------------------------------------------------- */

int MinPreconLBFGS::iterate(int maxiter) {
  if (!precon_ready_) setup_precon();

  for (int iter = 0; iter < maxiter; ++iter) {
    if (timer->check_timeout(niter)) return TIMEOUT;
    update->ntimestep = ++update->ntimestep;
    const bigint ntimestep = update->ntimestep;
    niter++;

    const int ne = nextra_global;

    // a neighbour rebuild may have migrated/re-sorted atoms -> reset history
    if (neighbor->ncalls != last_ncalls_) {
      s_hist_.clear();  y_hist_.clear();  rho_hist_.clear();
      sc_hist_.clear(); yc_hist_.clear(); rhoc_hist_.clear();
      has_prev_ = false;
      last_ncalls_ = neighbor->ncalls;
    }
    if (static_cast<int>(grad_.size()) != nvec) {
      grad_.assign(nvec, 0.0);
      g_prev_.assign(nvec, 0.0);
      x_prev_.assign(nvec, 0.0);
      s_hist_.clear();  y_hist_.clear();  rho_hist_.clear();
      sc_hist_.clear(); yc_hist_.clear(); rhoc_hist_.clear();
      has_prev_ = false;
    }
    if (static_cast<int>(gradc_.size()) != ne) {
      gradc_.assign(ne, 0.0);
      gc_prev_.assign(ne, 0.0);
    }

    for (int k = 0; k < nvec; ++k) grad_[k] = -fvec[k];
    for (int e = 0; e < ne; ++e) gradc_[e] = -fextra[e];

    // --- LBFGS history updates from the previous step --------------------
    if (has_prev_) {
      std::vector<double> s(nvec), y(nvec);
      for (int k = 0; k < nvec; ++k) {
        s[k] = xvec[k] - x_prev_[k];
        y[k] = grad_[k] - g_prev_[k];
      }
      const double ys = ddot(s.data(), y.data(), nvec);
      if (ys > EPS_YS) {
        s_hist_.push_back(std::move(s));
        y_hist_.push_back(std::move(y));
        rho_hist_.push_back(1.0 / ys);
        if (static_cast<int>(s_hist_.size()) > memory_) {
          s_hist_.erase(s_hist_.begin());
          y_hist_.erase(y_hist_.begin());
          rho_hist_.erase(rho_hist_.begin());
        }
      }
      if (ne > 0) {
        // cell displacement = alpha_final * hextra (both still hold the
        // previous iteration's values); the cell DOF have no stored coord
        std::vector<double> sc(ne), yc(ne);
        double ysc = 0.0;
        for (int e = 0; e < ne; ++e) {
          sc[e] = alpha_final * hextra[e];
          yc[e] = gradc_[e] - gc_prev_[e];
          ysc += sc[e] * yc[e];
        }
        if (ysc > EPS_YS) {
          sc_hist_.push_back(std::move(sc));
          yc_hist_.push_back(std::move(yc));
          rhoc_hist_.push_back(1.0 / ysc);
          if (static_cast<int>(sc_hist_.size()) > memory_) {
            sc_hist_.erase(sc_hist_.begin());
            yc_hist_.erase(yc_hist_.begin());
            rhoc_hist_.erase(rhoc_hist_.begin());
          }
        }
      }
    }
    for (int k = 0; k < nvec; ++k) {
      x_prev_[k] = xvec[k];
      g_prev_[k] = grad_[k];
    }
    for (int e = 0; e < ne; ++e) gc_prev_[e] = gradc_[e];
    has_prev_ = true;

    // --- atomic search direction: preconditioned LBFGS two-loop ----------
    const int m = static_cast<int>(s_hist_.size());
    std::vector<double> q(grad_);
    std::vector<double> alpha(m, 0.0);
    for (int k = m - 1; k >= 0; --k) {
      alpha[k] = rho_hist_[k] * ddot(s_hist_[k].data(), q.data(), nvec);
      for (int t = 0; t < nvec; ++t) q[t] -= alpha[k] * y_hist_[k][t];
    }
    fix_->assemble(fix_->mu());
    std::vector<double> z(nvec, 0.0);
    fix_->solve(q.data(), z.data());  // z = P_pos^-1 q (distributed Jacobi-CG)
    for (int k = 0; k < m; ++k) {
      const double b = rho_hist_[k] * ddot(y_hist_[k].data(), z.data(), nvec);
      for (int t = 0; t < nvec; ++t) z[t] += (alpha[k] - b) * s_hist_[k][t];
    }
    for (int k = 0; k < nvec; ++k) h[k] = -z[k];

    // --- cell search direction: separate LBFGS two-loop, H0 = 1/mu_c -----
    if (ne > 0) {
      const int mc = static_cast<int>(sc_hist_.size());
      std::vector<double> qc(gradc_);
      std::vector<double> alphac(mc, 0.0);
      for (int k = mc - 1; k >= 0; --k) {
        double sq = 0.0;
        for (int e = 0; e < ne; ++e) sq += sc_hist_[k][e] * qc[e];
        alphac[k] = rhoc_hist_[k] * sq;
        for (int e = 0; e < ne; ++e) qc[e] -= alphac[k] * yc_hist_[k][e];
      }
      const double mu_c = fix_->mu_c();
      std::vector<double> zc(ne);
      for (int e = 0; e < ne; ++e) zc[e] = qc[e] / mu_c;
      for (int k = 0; k < mc; ++k) {
        double yz = 0.0;
        for (int e = 0; e < ne; ++e) yz += yc_hist_[k][e] * zc[e];
        const double b = rhoc_hist_[k] * yz;
        for (int e = 0; e < ne; ++e) zc[e] += (alphac[k] - b) * sc_hist_[k][e];
      }
      for (int e = 0; e < ne; ++e) hextra[e] = -zc[e];
    }

    // line search (LAMMPS' MPI-safe backtracking; steps h and hextra)
    eprevious = ecurrent;
    const int fail = (this->*linemin)(ecurrent, alpha_final);
    if (fail) return fail;

    // convergence (fnorm_* include the cell DOF fextra)
    if (neval >= update->max_eval) return MAXEVAL;
    if (std::fabs(ecurrent - eprevious) <
        update->etol * 0.5 *
            (std::fabs(ecurrent) + std::fabs(eprevious) + EPS_ENERGY))
      return ETOL;
    if (update->ftol > 0.0) {
      double fdotf;
      if (normstyle == MAX) fdotf = fnorm_max();
      else if (normstyle == INF) fdotf = fnorm_inf();
      else fdotf = fnorm_sqr();
      if (fdotf < update->ftol * update->ftol) return FTOL;
    }

    if (output->next == ntimestep) {
      timer->stamp();
      output->write(ntimestep);
      timer->stamp(Timer::OUTPUT);
    }
  }
  return MAXITER;
}
