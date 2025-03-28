cmake_minimum_required(VERSION 3.14)
project(zseeker LANGUAGES CXX) # need this so CMAKE_CXX_COMPILER_ID var will be populated apparently

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/b796f7d44681514f58a683a3a71ff17c94edb0c1.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  project(zseeker LANGUAGES CUDA CXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconstexpr-ops-limit=1000000000 -O3")

  # I need the section below to avoid getting an error for empty CUDA_ARCHITECTURES
  # TODO: Eventually on CMake 3.23 I can just use 'all' or in 3.24 'native' instead
  # of this craziness. See https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
  include(FindCUDA/select_compute_arch)
  CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
  message(INSTALLED_GPU_CCS_1="${INSTALLED_GPU_CCS_1}") # only added this line to see what it is, can remove
  string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
  string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
  string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
  message(CUDA_ARCH_LIST="${CUDA_ARCH_LIST}") # only added this line to see what it is, can remove
  SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
  SET(CMAKE_CUDA_ARCHITECTURES 75) # hard-coding this to my local GPU arch so I can build my docker images for it, can remove later maybe?

add_executable(
	zseeker
	checkz3constantswithz5usingLUTandCPU.cpp
	cudawrapper.hpp
	checkz3constantswithz5usingLUTandGPU.cu
)

add_executable(
  zseeker2
  main.cpp
  PolynomialCheckerInterface.hpp
  CpuPolynomialChecker.hpp
  CpuQuinticLastChecker.cpp
  GpuPolynomialChecker.hpp
  GpuQuinticLastChecker.cu
  GpuQuinticFirstChecker.cu
)

add_executable(
  GpuPolynomialCheckerTestSuite
  test/GpuPolynomialCheckerTestSuite.cpp
  GpuQuinticLastChecker.cu
  GpuQuinticFirstChecker.cu
)
target_link_libraries(
  GpuPolynomialCheckerTestSuite
  GTest::gtest_main
)

# Uncomment the below out to see detailed register and memory information for CUDA compilation
#target_compile_options(GpuPolynomialCheckerTestSuite PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)

gtest_discover_tests(GpuPolynomialCheckerTestSuite)
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconstexpr-steps=1000000000 -O3")

  add_executable(
    zseeker
    checkz3constantswithz5usingLUTandCPU.cpp
    #cudawrapper.hpp
    #checkz3constantswithz5usingLUTandGPU.cu
  )

  add_executable(
    zseeker2
    main.cpp
    PolynomialCheckerInterface.hpp
    CpuPolynomialChecker.hpp
    CpuQuinticLastChecker.cpp
    #GpuPolynomialChecker.hpp
    #GpuQuinticLastChecker.cu
    #GpuQuinticFirstChecker.cu
  )
endif()

enable_testing()

add_executable(
  PrecisionTestSuite
  test/PrecisionTestSuite.cpp
)
target_link_libraries(
  PrecisionTestSuite
  GTest::gtest_main
)

add_executable(
  PrintPrecisions
  PrintPrecisions.cpp
)

add_executable(
  ArithmeticTestSuite
  test/ArithmeticTestSuite.cpp
)
target_link_libraries(
  ArithmeticTestSuite
  GTest::gtest_main
)

add_executable(
  LookupTableTestSuite
  test/LookupTableTestSuite.cpp
)
target_link_libraries(
  LookupTableTestSuite
  GTest::gtest_main
)

add_executable(
  CpuPolynomialCheckerTestSuite
  test/CpuPolynomialCheckerTestSuite.cpp
  CpuQuinticLastChecker.cpp
  CpuQuinticFirstChecker.cpp
  CpuQuinticFirstWithBreakoutsChecker.cpp
)
target_link_libraries(
  CpuPolynomialCheckerTestSuite
  GTest::gtest_main
)

gtest_discover_tests(LookupTableTestSuite)
gtest_discover_tests(PrecisionTestSuite)
gtest_discover_tests(ArithmeticTestSuite)
gtest_discover_tests(CpuPolynomialCheckerTestSuite)