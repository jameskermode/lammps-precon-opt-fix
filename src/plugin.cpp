/* ----------------------------------------------------------------------
   LAMMPS Plugin registration

   This file registers the fix with LAMMPS plugin system
------------------------------------------------------------------------- */

#include "lammpsplugin.h"
#include "version.h"
#include "fix_precon_lbfgs.h"

using namespace LAMMPS_NS;

// Creator function for the fix
static Fix *fix_precon_lbfgs_creator(LAMMPS *lmp, int argc, char **argv)
{
  return new FixPreconLBFGS(lmp, argc, argv);
}

// Register the plugin
extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  lammpsplugin_t plugin;
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc) regfunc;

  plugin.version = LAMMPS_VERSION;
  plugin.style = "fix";
  plugin.name = "precon_lbfgs";
  plugin.info = "Preconditioned LBFGS optimizer v1.0.0";
  plugin.author = "James Kermode";
  plugin.creator.v2 = (lammpsplugin_factory2 *) &fix_precon_lbfgs_creator;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);
}
