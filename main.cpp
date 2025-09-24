#include <iostream>
#include "CpuPolynomialChecker.hpp"
#include "GpuPolynomialChecker.hpp"
#include "math.hpp"
#include "lookupTable.hpp"

int main(int argc, char *argv[])
{
    PolynomialCheckerInterface *checker;
    std::vector<int*> *hits;
    long floatHitCount = 0;

    switch (argc) {
		case 3:
		{
            // TODO: Make this argument handling WAY more robust and intuitive
			checker = new CpuQuinticLastChecker();
            break;
		}
        case 2:
        {
            checker = new GpuQuinticFirstChecker();
            break;
        }
        default:
        {
            checker = new GpuQuinticLastChecker();
            break;
        }
	}

    hits = checker->findHits(ZETA5, M_PI, 5, LUT.data(), NULL, floatHitCount);
    std::cout << "Count=" << hits->size() << std::endl;
    std::cout << "OutputCount=" << floatHitCount << std::endl;
    int *result = hits->at(0);
    std::cout << "Vals=" << result[0] << "," << result[1] << "," << result[2] << "," << result[3] << "," << result[4] << "," << result[5] << "," << std::endl;
    // result = hits->at(27);
    // std::cout << "Vals=" << result[0] << "," << result[1] << "," << result[2] << "," << result[3] << "," << result[4] << "," << result[5] << "," << std::endl;
    
    // clean up
    checker->~PolynomialCheckerInterface();

    return 0;
}