cmake_minimum_required(VERSION 3.14)

project(Kernel
    VERSION 2.0.0
    DESCRIPTION "Quantum Kernel — Pipeline of Coherence"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Installation paths ───────────────────────────────────────────────────────
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# ── Header-only library ──────────────────────────────────────────────────────
add_library(Kernel INTERFACE)
add_library(Kernel::Kernel ALIAS Kernel)

target_include_directories(Kernel INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(Kernel INTERFACE cxx_std_17)

# ── Benchmarks ───────────────────────────────────────────────────────────────
option(KERNEL_BUILD_BENCHMARKS "Build benchmark executables" ON)

if(KERNEL_BUILD_BENCHMARKS)
    add_executable(benchmark_nist_ir8356 benchmark_nist_ir8356.cpp)
    target_link_libraries(benchmark_nist_ir8356 PRIVATE Kernel::Kernel m)

    find_package(OpenSSL QUIET)
    add_executable(benchmark_pow_nonce_search benchmark_pow_nonce_search.cpp)
    target_link_libraries(benchmark_pow_nonce_search PRIVATE Kernel::Kernel m)
    if(OpenSSL_FOUND)
        target_link_libraries(benchmark_pow_nonce_search PRIVATE OpenSSL::Crypto)
    else()
        message(WARNING "OpenSSL not found via find_package; falling back to system 'crypto' library for benchmark_pow_nonce_search")
        target_link_libraries(benchmark_pow_nonce_search PRIVATE crypto)
    endif()

    add_executable(benchmark_scaling_falsification benchmark_scaling_falsification.cpp)
    target_link_libraries(benchmark_scaling_falsification PRIVATE Kernel::Kernel m)
endif()

# ── Tests ────────────────────────────────────────────────────────────────────
option(KERNEL_BUILD_TESTS "Build test executables and register with CTest" ON)

if(KERNEL_BUILD_TESTS)
    enable_testing()

    set(_kernel_tests
        test_chiral_nonlinear_gate
        test_coherent_search
        test_kernel_pipeline
        test_interrupt_nist
        test_ipc
        test_noise_scaling_phase_transition
        test_ohm_coherence
        test_palindrome_precession
        test_pipeline_theorems
        test_qudit_kernel
    )

    foreach(_test IN LISTS _kernel_tests)
        add_executable(${_test} ${_test}.cpp)
        target_link_libraries(${_test} PRIVATE Kernel::Kernel m)
        add_test(NAME ${_test} COMMAND ${_test})
    endforeach()
endif()

# ── Installation ─────────────────────────────────────────────────────────────
set(_kernel_headers
    ChiralNonlinearGate.hpp
    KernelPipeline.hpp
    KernelState.hpp
    LadderChiralSearch.hpp
    LadderSearchBenchmark.hpp
    PalindromePrecession.hpp
    SpectralBridge.hpp
    ohm_coherence_duality.hpp
)

install(FILES ${_kernel_headers}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Kernel
)

install(TARGETS Kernel
    EXPORT KernelTargets
)

install(EXPORT KernelTargets
    FILE KernelTargets.cmake
    NAMESPACE Kernel::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Kernel
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/KernelConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/KernelConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Kernel
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/KernelConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/KernelConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/KernelConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Kernel
)
