//
// Created by iskakoff on 19/07/16.
//
#include <iostream>

#include <edlib/EDParams.h>
#include <cstdlib>
#include "edlib/Hamiltonian.h"
#include "edlib/SzSymmetry.h"
#include "edlib/SOCRSStorage.h"
#include "edlib/CRSStorage.h"
#include "edlib/HubbardModel.h"
#include "edlib/GreensFunction.h"
#include "edlib/ChiLoc.h"
#include "edlib/HDF5Utils.h"
#include "edlib/SpinResolvedStorage.h"
#include "edlib/StaticObservables.h"
#include "edlib/MeshFactory.h"
#include "sparse_matsubara.h"


void define_params(alps::params &params) {
  params.define < int >("single.NEV", 1, "Number of eigenvalues to find for single EV calculations.");
  params.define < int >("single.NCV", 7, "Number of convergent values for single EV calculations.");
}

int main(int argc, const char ** argv) {
// Init MPI if enabled
#ifdef USE_MPI
  MPI_Init(&argc, (char ***) &argv);
  alps::mpi::communicator comm;
#endif
// Define and read model parameters
  alps::params params(argc, argv);
  EDLib::define_parameters(params);
  define_params(params);
  params.define<bool>("REAL_FREQ", false, "Compute on real frequency grid.");
  params.define<std::string>("FREQ_FILE", "1e5_112.hdf5", "File with frequency grid.");
  params.define<std::string>("FREQ_PATH", "fermi/wsample", "File with frequency grid.");
  if(params.help_requested(std::cout)) {
    std::exit(0);
  }
  // open output file
  alps::hdf5::archive ar;
#ifdef USE_MPI
  if(!comm.rank())
#endif
    ar.open(params["OUTPUT_FILE"].as<std::string>(), "w");
// Start calculations
  try {
    // Construct Hamiltonian object
    typedef EDLib::SRSSIAMHamiltonian HType;
#ifdef USE_MPI
    HType ham(params, comm);
#else
    HType ham(params);
#endif
    // Diagonalize Hamiltonian
    if (!params["arpack.SECTOR"].as<bool>())
    {
      alps::params params2 = params;
      params2["arpack.NEV"] = 1;
      params2["arpack.NCV"] = params["single.NCV"];
      params2["storage.EIGENVALUES_ONLY"] = 1;
      params2["arpack.SECTOR"] = false;
      params["arpack.SECTOR"] = true;
#ifdef USE_MPI
      HType ham2(params2, comm);
#else
      HType ham2(params2);
#endif
      ham2.diag();
      std::queue<EDLib::Symmetry::SzSymmetry::Sector> sectors;
      std::vector<std::vector<int> > sectors_list;
      
      EDLib::Combination _comb(params["NSITES"]);
      const EDLib::EigenPair<double, EDLib::Symmetry::SzSymmetry::Sector> &gs = *ham2.eigenpairs().begin();
      for (auto kkk = ham2.eigenpairs().begin(); kkk != ham2.eigenpairs().end(); kkk++) {
        const EDLib::EigenPair<double, EDLib::Symmetry::SzSymmetry::Sector> &eigenpair = *kkk;
        if(std::exp(-(eigenpair.eigenvalue() - gs.eigenvalue()) * params["lanc.BETA"].as<double>()) > params["lanc.BOLTZMANN_CUTOFF"].as<double>() ) {
          sectors.push(eigenpair.sector());
        }
      }
      ham.model().symmetry().sectors() = sectors;
#ifdef USE_MPI
      MPI_Barrier(comm);
#endif
    }
    ham.diag();
    // Save eigenvalues to HDF5 file
    EDLib::hdf5::save_eigen_pairs(ham, ar, "results");
    // Construct Green's function object
    EDLib::gf::GreensFunction < HType, seet::SparseMeshFactory, alps::gf::statistics::statistics_type> greensFunction(params, ham,alps::gf::statistics::statistics_type::FERMIONIC);
    // Compute and save Green's function
    greensFunction.compute();
    greensFunction.save(ar, "results");

    if (params["REAL_FREQ"].as<bool>() ) {
      EDLib::gf::GreensFunction < HType, EDLib::RealFreqMeshFactory> greensFunction_r(params, ham);
      greensFunction_r.compute();
      greensFunction_r.save(ar, "results");
    }
    // Init two particle Green's function object
    //EDLib::gf::ChiLoc<HType, alps::gf::real_frequency_mesh> susc(params, ham);
    // Compute and save spin susceptibility
    //susc.compute();
    //susc.save(ar, "results");
    // Compute and save charge susceptibility
    //susc.compute<EDLib::gf::NOperator<double> >();
    //susc.save(ar, "results");
  } catch (std::exception & e) {
#ifdef USE_MPI
    if(comm.rank() == 0) {
      std::cerr<<e.what();
      ar.close();
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
#else
    std::cerr<<e.what();
#endif
  }
#ifdef USE_MPI
  if(!comm.rank())
#endif
  ar.close();
#ifdef USE_MPI
  MPI_Finalize();
#endif
  return 0;
}
