#include <iostream>
#include "CpuPolynomialChecker.hpp"
#include "GpuPolynomialChecker.hpp"
#include "math.hpp"

int main(int argc, char *argv[])
{
    PolynomialCheckerInterface *checker;
    std::vector<float> *hits;

    switch (argc) {
		case 2:
		{
            // TODO: Make this argument handling WAY more robust and intuitive
			checker = new CpuPolynomialChecker();
            break;
		}
        default:
        {
            checker = new GpuPolynomialChecker();
            break;
        }
	}

    hits = checker->findHits(M_PI, ZETA4, NULL);
    std::cout << "Result=" << hits->at(0) << std::endl;
    
    return 0;
}