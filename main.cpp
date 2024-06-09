#include "CpuPolynomialChecker.hpp"
#include "GpuPolynomialChecker.hpp"

int main(int argc, char *argv[])
{
    PolynomialCheckerInterface *checker;

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

    checker->findHits();

    return 0;
}