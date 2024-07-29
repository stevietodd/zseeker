#include <iostream>
#include "CpuPolynomialChecker.hpp"
#include "GpuPolynomialChecker.hpp"
#include "math.hpp"
#include "lookupTable.hpp"

int main(int argc, char *argv[])
{
    PolynomialCheckerInterface *checker;
    std::vector<int*> *hits;

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

    hits = checker->findHits(ZETA5, M_PI, 5, LUT.data(), NULL);
    std::cout << "Count=" << hits->size() << std::endl;
    int *result = hits->at(0);
    std::cout << "Vals=" << result[0] << "," << result[1] << "," << result[2] << "," << result[3] << "," << result[4] << "," << result[5] << "," << std::endl;
    // result = hits->at(27);
    // std::cout << "Vals=" << result[0] << "," << result[1] << "," << result[2] << "," << result[3] << "," << result[4] << "," << result[5] << "," << std::endl;
    
    // clean up
    checker->~PolynomialCheckerInterface();

    return 0;
}