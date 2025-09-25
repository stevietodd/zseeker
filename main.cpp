#include <iostream>
#include <string>
#include <strings.h>
#include "CpuPolynomialChecker.hpp"
#include "GpuPolynomialChecker.hpp"
#include "math.hpp"
#include "lookupTable.hpp"

int main(int argc, char *argv[])
{
    PolynomialCheckerInterface *checker;
    std::vector<int*> *hits;
    long floatHitCount = 0;
	double theConst;

    switch (argc) {
		case 3:
		{
            std::string constString(argv[1]); // Convert second argument to std::string
			try {
				theConst = std::stod(constString);
				std::cout << "Using constant " << theConst << std::endl;
			} catch (const std::invalid_argument& e) {
				std::cerr << "Error: Invalid argument for constant - " << e.what() << std::endl;
				std::cout << "EXITING" << std::endl;
				return 0;
			} catch (const std::out_of_range& e) {
				std::cerr << "Error: Out of range constant - " << e.what() << std::endl;
				std::cout << "EXITING" << std::endl;
				return 0;
			}

			if (strcasecmp(argv[2], "gl") == 0) {
				std::cout << "Using GpuQuinticLast" << std::endl;
				checker = new GpuQuinticLastChecker();
			} else if (strcasecmp(argv[2], "gf") == 0) {
				std::cout << "Using GpuQuinticFirst" << std::endl;
				checker = new GpuQuinticFirstChecker();
			} else if (strcasecmp(argv[2], "cl") == 0) {
				std::cout << "Using CpuQuinticLast" << std::endl;
				checker = new CpuQuinticLastChecker();
			} else if (strcasecmp(argv[2], "cf") == 0) {
				std::cout << "Using CpuQuinticFirst" << std::endl;
				checker = new CpuQuinticFirstChecker();
			} else if (strcasecmp(argv[2], "cfwb") == 0) {
				std::cout << "Using CpuQuinticFirstWithBreakouts" << std::endl;
				checker = new CpuQuinticFirstWithBreakoutsChecker();
			} else {
				std::cout << "Could not parse checker, using CpuQuinticFirstWithBreakouts" << std::endl;
				checker = new CpuQuinticFirstWithBreakoutsChecker();
			}

			break;
		}
        case 2:
        {
			std::cout << "No checker specified: using CpuQuinticFirstWithBreakouts" << std::endl;
            checker = new CpuQuinticFirstWithBreakoutsChecker();

			std::string constString(argv[1]); // Convert second argument to std::string
			try {
				theConst = std::stod(constString);
				std::cout << "Using constant " << theConst << std::endl;
			} catch (const std::invalid_argument& e) {
				std::cerr << "Error: Invalid argument for constant - " << e.what() << std::endl;
				std::cout << "EXITING" << std::endl;
				return 0;
			} catch (const std::out_of_range& e) {
				std::cerr << "Error: Out of range constant - " << e.what() << std::endl;
				std::cout << "EXITING" << std::endl;
				return 0;
			}

            break;
        }
        default:
        {
			std::cout << "No arguments provided: assuming constant PI using CpuQuinticFirstWithBreakouts" << std::endl;
			theConst = M_PI;
			checker = new CpuQuinticFirstWithBreakoutsChecker();
            break;
        }
	}

    hits = checker->findHits(ZETA5, theConst, 5, LUT.data(), NULL, floatHitCount);
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