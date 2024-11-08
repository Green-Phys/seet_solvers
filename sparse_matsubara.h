#ifndef SPARSE_MATSUBARA_H
#define SPARSE_MATSUBARA_H


#ifdef ALPS_HAVE_MPI
#include <alps/gf/mpi_bcast.hpp>
#endif
#include <alps/gf/mesh/index.hpp>
#include <alps/gf/mesh/mesh_base.hpp>

namespace seet {
  class sparse_matsubara_mesh : public alps::gf::base_mesh {
    double beta_;
    int nfreq_;

    alps::gf::statistics::statistics_type statistics_;

    inline void throw_if_empty() const {
      if (extent() == 0) {
        throw std::runtime_error("matsubara_mesh is empty");
      }
    }

  public:
    typedef alps::gf::generic_index<sparse_matsubara_mesh> index_type;
    /// copy constructor
    sparse_matsubara_mesh(const sparse_matsubara_mesh& rhs) : alps::gf::base_mesh(rhs), beta_(rhs.beta_), nfreq_(rhs.nfreq_), statistics_(rhs.statistics_) {check_range();}
    sparse_matsubara_mesh():
        beta_(0.0), nfreq_(0), statistics_(alps::gf::statistics::FERMIONIC)
    {
    }

    sparse_matsubara_mesh(double b, const std::vector<double> &points, alps::gf::statistics::statistics_type statistics=alps::gf::statistics::FERMIONIC):
        beta_(b), nfreq_(points.size()), statistics_(statistics) {
      _points().resize(nfreq_);
      std::copy(points.begin(), points.end(), _points().begin());
      check_range();
    }
    int extent() const{return nfreq_;}


    int operator()(index_type idx) const {
#ifndef NDEBUG
      throw_if_empty();
#endif
      return idx();
    }

    /// Comparison operators
    bool operator==(const sparse_matsubara_mesh &mesh) const {
      throw_if_empty();
      return beta_==mesh.beta_ && nfreq_==mesh.nfreq_ && statistics_==mesh.statistics_;
    }

    /// Comparison operators
    bool operator!=(const sparse_matsubara_mesh &mesh) const {
      throw_if_empty();
      return !(*this==mesh);
    }

    ///getter functions for member variables
    double beta() const{ return beta_;}
    alps::gf::statistics::statistics_type statistics() const{ return statistics_;}

    /// Swaps this and another mesh
    // It's a member function to avoid dealing with templated friend decalration.
    void swap(sparse_matsubara_mesh& other) {
      throw_if_empty();
      if(statistics_!=other.statistics_)
        throw std::runtime_error("Attempt to swap two meshes with different statistics.");// FIXME: specific exception
      std::swap(this->beta_, other.beta_);
      std::swap(this->nfreq_, other.nfreq_);
      base_mesh::swap(other);
    }

    void save(alps::hdf5::archive& ar, const std::string& path) const
    {
      throw_if_empty();
      ar[path+"/kind"] << "MATSUBARA";
      ar[path+"/N"] << nfreq_;
      ar[path+"/statistics"] << int(statistics_); //
      ar[path+"/beta"] << beta_;
      ar[path+"/points"] << points();
    }

    void load(alps::hdf5::archive& ar, const std::string& path)
    {
      std::string kind;
      ar[path+"/kind"] >> kind;
      if (kind!="SPARSE_MATSUBARA") throw std::runtime_error("Attempt to read Matsubara mesh from non-Matsubara data, kind="+kind); // FIXME: specific exception
      double nfr, beta;
      int stat, posonly;

      ar[path+"/N"] >> nfr;
      ar[path+"/statistics"] >> stat;
      ar[path+"/beta"] >> beta;
      ar[path+"/points"] >> _points();

      statistics_=alps::gf::statistics::statistics_type(stat);
      beta_=beta;
      nfreq_=nfr;
      check_range();
    }

    /// Save to HDF5
    void save(alps::hdf5::archive& ar) const
    {
      save(ar, ar.get_context());
    }

    /// Load from HDF5
    void load(alps::hdf5::archive& ar)
    {
      load(ar, ar.get_context());
    }

#ifdef ALPS_HAVE_MPI
    void broadcast(const alps::mpi::communicator& comm, int root)
    {
    }
#endif

    void check_range(){
      if(statistics_!=alps::gf::statistics::FERMIONIC && statistics_!=alps::gf::statistics::BOSONIC) throw std::invalid_argument("statistics should be bosonic or fermionic");
      throw_if_empty();
    }

  };
  ///Stream output operator, e.g. for printing to file
  std::ostream &operator<<(std::ostream &os, const sparse_matsubara_mesh &M){
    os<<"# "<<"MATSUBARA"<<" mesh: N: "<<M.extent()<<" beta: "<<M.beta()<<" statistics: ";
    os<<(M.statistics()==alps::gf::statistics::FERMIONIC?"FERMIONIC":"BOSONIC")<<" ";
    os<<std::endl;
    return os;
  }

  /// Swaps two Matsubara meshes
  void swap(sparse_matsubara_mesh& a, sparse_matsubara_mesh& b) {
    a.swap(b);
  }

  class SparseMeshFactory {
  public:
    using MeshType = sparse_matsubara_mesh;
    static MeshType createMesh(alps::params &p, alps::gf::statistics::statistics_type type) {
      std::vector<double> omegas;
      std::vector<int> ns;
      std::string file = p["FREQ_FILE"].as<std::string>();
      std::string path = p["FREQ_PATH"].as<std::string>();
      double beta = p["lanc.BETA"];
      alps::hdf5::archive ar(file, "r");
      ar[path] >> ns;
      for(auto n : ns) {
        omegas.push_back((2*n+1)*M_PI/beta);
      }
      return std::move(sparse_matsubara_mesh(beta, omegas, type));
    }
  };
}
#endif