#ifndef GPU_POLYNOMIAL_CHECKER_HPP
#define GPU_POLYNOMIAL_CHECKER_HPP

#include "PolynomialCheckerInterface.hpp"

class GpuPolynomialChecker : public PolynomialCheckerInterface
{
    public:
        GpuPolynomialChecker(){}
        ~GpuPolynomialChecker(){}
        std::vector<float>* findHits(const float theConst, const float needle, const std::vector<float> *coeffArray);
};

#endif // GPU_POLYNOMIAL_CHECKER_HPP