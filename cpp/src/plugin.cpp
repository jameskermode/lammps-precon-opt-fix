/* Stage 8 — plugin registration.

   Registers `fix precon/exp` and `min_style precon/lbfgs` with a LAMMPS
   instance that loads this shared library via the `plugin` command.
*/
#include "lammpsplugin.h"
#include "version.h"

#include "fix_precon_exp.h"
#include "min_precon_lbfgs.h"

using namespace LAMMPS_NS;

static Min *min_precon_lbfgs_creator(LAMMPS *lmp) {
  return new MinPreconLBFGS(lmp);
}

static Fix *fix_precon_exp_creator(LAMMPS *lmp, int argc, char **argv) {
  return new FixPreconExp(lmp, argc, argv);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc) {
  lammpsplugin_t plugin;
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc) regfunc;

  plugin.version = LAMMPS_VERSION;
  plugin.handle = handle;

  // fix precon/exp — neighbour-list provider for the preconditioner
  plugin.style = "fix";
  plugin.name = "precon/exp";
  plugin.info = "Exp preconditioner helper fix (lammps-precon-opt Stage 8)";
  plugin.author = "lammps-precon-opt";
  plugin.creator.v2 = (lammpsplugin_factory2 *) &fix_precon_exp_creator;
  (*register_plugin)(&plugin, lmp);

  // min_style precon/lbfgs — preconditioned LBFGS minimizer
  plugin.style = "min";
  plugin.name = "precon/lbfgs";
  plugin.info = "Preconditioned LBFGS minimizer (lammps-precon-opt Stage 8)";
  plugin.author = "lammps-precon-opt";
  plugin.creator.v1 = (lammpsplugin_factory1 *) &min_precon_lbfgs_creator;
  (*register_plugin)(&plugin, lmp);
}
