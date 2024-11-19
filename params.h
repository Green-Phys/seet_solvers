//
// Created by katherlee on 2020-04-20.
//

#ifndef _PARAMS_H_
#define _PARAMS_H_

#include <args.hxx>
#include <iniparser.hpp>
#include <iostream>

#include "gfmol/sim_modes.h"

namespace gfmol {

inline bool exists(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

struct Params {
  Mode mode;
  bool unrestricted;
  bool decomposed;
  std::string repr;
  std::string repr_file;
  std::string lambda;
  double decomp_prec;
  double etol;
  int maxiter;
  int ncoeff;
  double beta;
  double damping;
  std::string hf_input;
  std::string output;
  bool checkpoint;
  std::string checkpoint_file;

  void validate() const
  {
    if (repr != "cheb" && repr != "ir")
      throw std::runtime_error("parameter repr: invalid repr " + repr);

    if (maxiter < 0)
      throw std::runtime_error("parameter maxiter: invalid value " + std::to_string(maxiter));

    if (beta <= 0)
      throw std::runtime_error("parameter beta: invalid value " + std::to_string(beta));

    if (damping < 0 || damping > 1)
      throw std::runtime_error("parameter damping: invalid value " + std::to_string(damping));
  }
};

inline std::ostream &operator<<(std::ostream &os, const Params &p)
{
  os << std::boolalpha;
  os << "Parameters:" << std::endl;
  os << "    mode:            " << (p.mode == Mode::GF2 ? "GF2" : "GW") << std::endl;
  os << "    unrestricted:    " << p.unrestricted << std::endl;
  os << "    decomposed:      " << p.decomposed << std::endl;
  os << "    repr:            " << p.repr << std::endl;
  os << "    repr_file:       " << p.repr_file << std::endl;
  os << "    decomp_prec:     " << p.decomp_prec << std::endl;
  os << "    etol:            " << p.etol << std::endl;
  os << "    maxiter:         " << p.maxiter << std::endl;
  os << "    beta:            " << p.beta << std::endl;
  os << "    damping:         " << p.damping << std::endl;
  os << "    hf_input:        " << p.hf_input << std::endl;
  os << "    output:          " << p.output << std::endl;
  os << "    checkpoint:      " << p.checkpoint << std::endl;
  os << "    checkpoint_file: " << p.checkpoint_file << std::endl;
  os << std::noboolalpha;
  return os;
}

// From Sergei
template <typename T>
T extract_value(const INI::File &f, args::ValueFlag<T> &parameter)
{
  return parameter ? parameter.Get() : f.GetValue(parameter.Name(), parameter.GetDefault()).template Get<T>();
}

template<typename E>
E extract_value(const INI::File & f, args::MapFlag<std::string, E> & parameter) {
  if(parameter) {
    return parameter.Get();
  }
  std::string value = f.GetValue(parameter.Name(), "").AsString();
  return value == "" ?  parameter.GetDefault(): parameter.GetMap().count(value) > 0 ? parameter.GetMap().find(value)->second : throw std::logic_error("No value in enum map");
};

inline bool extract_value(const INI::File &f, args::Flag &parameter)
{
  return parameter ? parameter.Get() : f.GetValue(parameter.Name(), false).Get<bool>();
}

inline Params parse_args(const int argc, char *const *const argv)
{
  std::unordered_map<std::string, Mode> solver_modes_map = {{"GW" , Mode::GW},
                                                            {"GF2", Mode::GF2}};

  args::ArgumentParser parser("gfmol: Green's function methods for molecules");
  args::Positional<std::string> inifile(parser, "inifile", "The parameter file");

  // Argument list
  args::MapFlag<std::string, Mode> mode(parser, "mode", "Type of calculations. GF2 or GW", {"mode"}, solver_modes_map, Mode::GF2);
  args::Flag unrestricted(parser, "unrestricted", "use unrestricted spin if set", {"unrestricted"});
  args::Flag decomposed(parser, "decomposed", "use unrestricted spin if set", {"decomposed"});
  args::ValueFlag<std::string> repr(parser, "repr", "type of representation: cheb or ir", {"repr"}, "cheb");
  args::ValueFlag<std::string> repr_file(parser, "repr-file", "path to representation file", {"repr-file"});
  args::ValueFlag<std::string> lambda(parser, "lambda", "path to representation file", {"lambda"}, "1e5");
  args::ValueFlag<double> decomp_prec(parser, "decomp-prec", "decomposition precision", {"decomp-prec"}, 1e-12);
  args::ValueFlag<double> etol(parser, "etol", "energy convergence tolarence", {"etol"}, 1e-8);
  args::ValueFlag<int> maxiter(parser, "maxiter", "maximum number of iterations", {"maxiter"}, 50);
  args::ValueFlag<int> ncoeff(parser, "ncoeff", "number of coefficient in nonuniform grid", {"ncoeff"}, 112);
  args::ValueFlag<double> beta(parser, "beta", "inverse temperature", {"beta"}, 100);
  args::ValueFlag<double> damping(parser, "damping", "damping factor", {"damping"});
  args::ValueFlag<std::string> hf_input(parser, "hf-input", "input hdf5 file", {"hf-input"});
  args::ValueFlag<std::string> output(parser, "output", "path to output HDF5 file", {"output"}, "gfmol-output.h5");
  args::Flag checkpoint(parser, "checkpoint", "use checkpoint file", {"checkpoint"});
  args::ValueFlag<std::string> checkpoint_file(parser,
                                               "checkpoint-file",
                                               "path to checkpoint HDF5 file",
                                               {"checkpoint-file"},
                                               "gfmol-chk.h5");

  args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});

  try {
    parser.ParseCLI(argc, argv);
  } catch (const args::Help &) {
    std::cout << parser;
    exit(0);
  }

  INI::File param_file;
  param_file.Load(inifile.Get(), true);

  Params p{extract_value(param_file, mode),
           extract_value(param_file, unrestricted),
           extract_value(param_file, decomposed),
           extract_value(param_file, repr),
           extract_value(param_file, repr_file),
           extract_value(param_file, lambda),
           extract_value(param_file, decomp_prec),
           extract_value(param_file, etol),
           extract_value(param_file, maxiter),
           extract_value(param_file, ncoeff),
           extract_value(param_file, beta),
           extract_value(param_file, damping),
           extract_value(param_file, hf_input),
           extract_value(param_file, output),
           extract_value(param_file, checkpoint),
           extract_value(param_file, checkpoint_file)};
  try {
    p.validate();
  } catch (const std::runtime_error &e) {
    std::cerr << "Error encountered! " << e.what() << std::endl;
    std::cout << parser;
    throw e;
  }
  return p;
}

} // namespace gfmol

#endif //_PARAMS_H_
