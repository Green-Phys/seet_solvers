cmake_minimum_required(VERSION 3.15)
project(EDLibExamples LANGUAGES C CXX Fortran
        VERSION 2.0)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set (CMAKE_CXX_STANDARD 11)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

find_package(ALPSCore COMPONENTS params hdf5 gf REQUIRED)
find_package(ARPACK REQUIRED)
find_package(BLAS)
find_package(LAPACK)
find_package(Eigen3 REQUIRED)
find_package(MPI REQUIRED)



option(USE_MPI "Use MPI for ED" OFF)
if(USE_MPI)
find_package(MPI REQUIRED)
endif(USE_MPI)
find_package(EDLib REQUIRED)
if(USE_MPI)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_MPI")
endif(USE_MPI)
find_package(Eigen3 REQUIRED)

set(extlibs
    ${Boost_LIBRARIES}
    ${HDF5_LIBRARIES}
    ${ALPSCore_LIBRARIES}
    ${ARPACK_LIB}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${PARPACK_LIB}
    ${MPI_CXX_LIBRARIES}
    ${MPI_Fortran_LIBRARIES}
    EDLib::EDLib EDLib::common-lib
    )

if(USE_MPI)
  set(parlibs
    ${PARPACK_LIB}
    ${MPI_CXX_LIBRARIES}
    ${MPI_Fortran_LIBRARIES}
    )
endif(USE_MPI)
    

add_executable(ed_solver Anderson.cpp)

target_link_libraries(ed_solver ${extlibs} Eigen3::Eigen)

if(USE_MPI)
  target_link_libraries(ed_solver ${parlibs})
endif(USE_MPI)

