
function(add_edlib_dependency)
    Include(FetchContent)

    FetchContent_Declare(
        EDLib
        GIT_REPOSITORY https://github.com/Green-Phys/EDLib.git
        GIT_TAG origin/master # or a later release
        CMAKE_ARGS
            -DARPACK_ROOT=${ARPACK_DIR}
    )

    FetchContent_MakeAvailable(EDLib)
endfunction()