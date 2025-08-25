#include "CpuPolynomialChecker.hpp"
#include "lookupTable.hpp"
#include "math.hpp"
#include <ctime> // can remove if getCurrentTimeString is removed
#include <iostream> // can remove if not using cout

char* getCurrentTimeString() {
	std::time_t currTime = std::time(nullptr);
	return std::asctime(std::localtime(&currTime));
}

std::vector<int*>* CpuQuinticLastChecker::findHits(
            const float needle,
            const float theConst,
            const int degree,
            const float *coeffArray,
            const std::vector<int> *loopRanges
)
{
    // make sure to remove these if I eventually stop using cout
    typedef std::numeric_limits< float > ldbl;
	std::cout.precision(ldbl::max_digits10);

    // TODO! Use coeffArray instead of LUT directly!!

    // Updated loop boundaries to match checkz3constantswithz5usingLUTandCPU::main
    // note that these are now negative to positive ranges instead of starting from 6
    int loopStartEnds[12] = {-608'383, 608'383, -152'231, 152'231, -6'087, 6'087, -2'203, 2'203, -555, 555, -143, 143};

    //TODO: Use degree for way more things than just processing loopRanges
    // if loopRanges is non-null, find first level with positive values (-1 indicates use default) and use those
    // (WRONG) note that we ignore any level after that since we don't want to skip coeffs in later loops (WRONG)
    // note that we DO allow all levels to be updated now but warn the user that they may have an incomplete search
    if (loopRanges != NULL) {
        // loopRanges must have (2*(degree+1)) elements. Format is [zStart, zEnd, yStart, yEnd, ...]
        for (int loopRangeInd = 0; loopRangeInd < (2*(degree+1)); loopRangeInd++) {
            //TODO: Make this not so hacky and stupid
            if (loopRanges->at(loopRangeInd) < USE_DEFAULT) {
                // they are setting a non-default value, so update loopStartEnds
                loopStartEnds[loopRangeInd] = loopRanges->at(loopRangeInd);
                std::cout << "WARNING: You have set a non-standard loop range. Your search may be incomplete" << std::endl;
            }
        }
    }

    const float theConst2 = powl(theConst, (float)2);
	const float theConst3 = powl(theConst, (float)3);
	const float theConst4 = powl(theConst, (float)4);
	const float theConst5 = powl(theConst, (float)5);

    float v0, v1, v2, v3, v4;
    int *hit;

    std::vector<int*> *hits = new std::vector<int*>();

    // Updated loops to use new boundaries and LUT access pattern
    for (int z = loopStartEnds[10]; z <= loopStartEnds[11]; z++) {
		v0 = (z < 0) ? -LUT[-z] : LUT[z];
	
		for (int y = loopStartEnds[8]; y <= loopStartEnds[9]; y++) {
			v1 = v0 + ((y < 0) ? -LUT[-y] : LUT[y]) * theConst;

			for (int x = loopStartEnds[6]; x <= loopStartEnds[7]; x++) {
				v2 = v1 + ((x < 0) ? -LUT[-x] : LUT[x]) * theConst2;

				for (int w = loopStartEnds[4]; w <= loopStartEnds[5]; w++) {
					v3 = v2 + ((w < 0) ? -LUT[-w] : LUT[w]) * theConst3;
                    //std::cout << "w=" << w << ", " << getCurrentTimeString();

                    for (int v = loopStartEnds[2]; v <= loopStartEnds[3]; v++) { // took 46-48 minutes per w-loop but gets results
                        v4 = v3 + ((v < 0) ? -LUT[-v] : LUT[v]) * theConst4;

                        for (int u = loopStartEnds[0]; u <= loopStartEnds[1]; u++) {
                            // note that we don't use a v5 variable anymore and compare directly to (needle - v4) to
                            // mimic how the Gpu checker does it

                            if (FLOAT_BASICALLY_EQUAL_DEFAULT(((u < 0) ? -LUT[-u] : LUT[u]) * theConst5, (needle - v4))) {
                                //printf("LUT[this]=%10.10lf,theConst5=%10.10lf,needle=%10.10lf,v4=%10.10lf,(needle-v4)=%10.10lf,diff=%10.10lf\n", LUT[u], theConst5, needle, v4, (needle-v4), ((LUT[u] * theConst5) - (needle-v4)));
                                hit = new int[6] {u, v, w, x, y, z};
                                hits->push_back(hit);
                                printHit(LUT.data(), u,v,w,x,y,z);
                            }
                        }
                    }
				}
			}
		}
	}

    return hits;
}