#include "params.h"
#include "gfmol/repn.h"
#include "gfmol/sim.h"
#include "gfmol/utils.h"
#include <iostream>

template<typename Repr>
void run(const gfmol::Params & p) {
  // Load input files
  h5e::File input_f(p.hf_input);
  gfmol::HartreeFock hf(input_f);
  gfmol::Mode mode = p.mode;
  h5e::File fout(p.output, h5e::File::Overwrite | h5e::File::ReadWrite | h5e::File::Create);
  // load repr
  h5e::File repr_f(p.repr_file);
  Repr frepr(repr_f, "/fermi", gfmol::Stats::Fermi, p.beta);
  Repr brepr(repr_f, "/bose", gfmol::Stats::Bose, p.beta);

  gfmol::SpinSimulation<Repr> p_sim(hf, frepr, brepr, p.mode, p.damping);

  gfmol::DTensor<4> gf(2, p_sim.frepr().ntau(), hf.nao(), hf.nao());
  gfmol::DTensor<4> sigma_t(2, p_sim.frepr().ntau(), hf.nao(), hf.nao());
  gfmol::DTensor<3> rho(2, hf.nao(), hf.nao());
  gfmol::DTensor<3> sigma_1(2, hf.nao(), hf.nao());

  sigma_t.set_zero();
  sigma_1.set_zero();

  gfmol::DTensor<3> gf_spin(hf.nao(), hf.nao(), p_sim.frepr().ntau());
  gfmol::DTensor<2> rho_spin(hf.nao(), hf.nao());
  std::cout<<"read data"<<std::endl;
  for (int is = 0; is < 2; ++is) {
    gf_spin = h5e::load<gfmol::DTensor<3>>(input_f, "gf/ftau/" + std::to_string(is+1));
    rho_spin = h5e::load<gfmol::DTensor<2>>(input_f, "rho/" + std::to_string(is+1));
    for (int ii = 0; ii < gf.shape()[2]; ++ii) {
      for (int ij = 0; ij < gf.shape()[3]; ++ij) {
        for (int it = 0; it < gf.shape()[1]; ++it) {
          gf(is, it, ii, ij) = gf_spin(ij, ii, it);
        }
        rho(is, ii, ij) = rho_spin(ij, ii);
      }
    }

  }
  std::cout<<"solve"<<std::endl;
  p_sim.p_sigma()->solve(sigma_t, gf);
  p_sim.p_sigma()->solve_HF(sigma_1, rho);
//  p_sigma_->solve_HF(sigmahf_, rho_);
//  p_sigma_->solve(sigmatau_, gtau_);

//  p_sim.run(p.maxiter, p.etol, &input_f);
//  p_sim.save(fout, "");

  for (int is = 0; is < 2; ++is) {
    for (int ii = 0; ii < gf.shape()[2]; ++ii) {
      for (int ij = 0; ij < gf.shape()[3]; ++ij) {
        for (int it = 0; it < gf.shape()[1]; ++it) {
          gf_spin(ij, ii, it) = sigma_t(is, it, ii, ij);
        }
        rho_spin(ij, ii) = sigma_1(is, ii, ij);
      }
    }
    h5e::dump(fout, "output/sigma1/" + std::to_string(is+1), rho_spin);
    h5e::dump(fout, "output/sigma2/" + std::to_string(is+1), gf_spin);
  }
}

int main(const int argc, char *const *const argv)
{
  using namespace gfmol;

  Params p = parse_args(argc, argv);

  std::cout << p << std::endl;
  if (p.repr == "cheb") {
    std::cout << "Loading Chebyshev repr at " << p.repr_file << std::endl;
    run<ChebyshevRepr>(p);
  } else if (p.repr == "ir") {
    std::cout << "Loading IR repr at " << p.repr_file << std::endl;
    run<IntermediateRepr>(p);
  }

  return 0;
}
