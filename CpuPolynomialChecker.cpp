#include "PolynomialCheckerInterface.hpp"

class CpuPolynomialChecker : public PolynomialCheckerInterface
{
    CpuPolynomialChecker(){}
    ~CpuPolynomialChecker(){}
    void findHits();
}