#ifndef GPU_POLYNOMIAL_CHECKER_HPP
#define GPU_POLYNOMIAL_CHECKER_HPP

#include "PolynomialCheckerInterface.hpp"

class GpuQuinticLastChecker : public PolynomialCheckerInterface
{
    public:
        GpuQuinticLastChecker(){}
        ~GpuQuinticLastChecker(){}

        std::vector<int*>* findHits(
            const float needle,
            const float theConst,
            const int degree,
            const float *coeffArray,
            const std::vector<int> *loopRanges
        );
};

class GpuQuinticFirstChecker : public PolynomialCheckerInterface
{
    public:
        GpuQuinticFirstChecker(){}
        ~GpuQuinticFirstChecker(){}

        std::vector<int*>* findHits(
            const float needle,
            const float theConst,
            const int degree,
            const float *coeffArray,
            const std::vector<int> *loopRanges
        );
};

#endif // GPU_POLYNOMIAL_CHECKER_HPP