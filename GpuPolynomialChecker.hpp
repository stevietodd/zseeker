#ifndef GPU_POLYNOMIAL_CHECKER_HPP
#define GPU_POLYNOMIAL_CHECKER_HPP

#include "PolynomialCheckerInterface.hpp"

class GpuPolynomialChecker : public PolynomialCheckerInterface
{
    public:
        GpuPolynomialChecker(){}
        ~GpuPolynomialChecker(){}
        std::vector<float>* findHits();
};

#endif // GPU_POLYNOMIAL_CHECKER_HPP