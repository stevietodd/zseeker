#ifndef GPU_OUTER_FIRST_POLYNOMIAL_CHECKER_HPP
#define GPU_OUTER_FIRST_POLYNOMIAL_CHECKER_HPP

#include "PolynomialCheckerInterface.hpp"

class GpuOuterFirstPolynomialChecker : public PolynomialCheckerInterface
{
    public:
        GpuOuterFirstPolynomialChecker(){}
        ~GpuOuterFirstPolynomialChecker(){}

        std::vector<int*>* findHits(
            const float needle,
            const float theConst,
            const int degree,
            const float *coeffArray,
            const std::vector<int> *loopRanges
        );
};

#endif // GPU_OUTER_FIRST_POLYNOMIAL_CHECKER_HPP