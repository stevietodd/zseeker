#include "PolynomialCheckerInterface.hpp"

class CpuPolynomialChecker : public PolynomialCheckerInterface
{
    CpuPolynomialChecker(){}
    ~CpuPolynomialChecker(){}
    std::vector<float> findHits();
};