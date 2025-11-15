#include "CpuPolynomialChecker.hpp"
#include "lookupTable.hpp"
#include "math.hpp"
#include <ctime> // can remove if getCurrentTimeString is removed
#include <iostream> // can remove if not using cout
#include <algorithm>
/*
TODO: Hackily removing these for now so tests will compile

char* getCurrentTimeString() {
	std::time_t currTime = std::time(nullptr);
	return std::asctime(std::localtime(&currTime));
}

void printHit(int i5, int i4, int i3, int i2, int i1, int i0)
{
	std::cout << "(" << i5 << "," << i4 << "," << i3 << "," << i2 << "," << i1 << "," << i0 << "): " <<
		LUT[i5] << "c^5 + " << LUT[i4] << "c^4 + " << LUT[i3] << "c^3 + " << LUT[i2] << "c^2 + " <<
		LUT[i1] << "c + " << LUT[i0] << " = HIT!\n";
}
*/

std::vector<int*>* CpuQuinticFirstChecker::findHits(
            const double needle,
            const double theConst,
            const int degree,
            const float *coeffArray,
            const std::vector<int> *loopRanges,
            long& floatHitCount
)
{
    // make sure to remove these if I eventually stop using cout
    typedef std::numeric_limits< float > ldbl;
	std::cout.precision(ldbl::max_digits10);

	// Leaving this printf in to help prevent floatHitCount from being optimized out
	//printf("CpuQuinticFirstChecker: floatHitCount starting as %ld\n", floatHitCount);

    // TODO! Use coeffArray instead of LUT directly!!

    // Updated loop boundaries to go from negative to positive ranges instead of starting from 6
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

    float maxValue = 0, floatTol = FLOAT_POS_ERROR_DEFAULT;
	double doubleValue = 0, doubleTol = DOUBLE_POS_ERROR_DEFAULT;
	const double theConst2 = powl(theConst, (double)2);
	const double theConst3 = powl(theConst, (double)3);
	const double theConst4 = powl(theConst, (double)4);
	const double theConst5 = powl(theConst, (double)5);
	const float needlef = (float)needle;
	const float theConstf = (float)theConst;
	const float theConst2f = (float)theConst2;
	const float theConst3f = (float)theConst3;
	const float theConst4f = (float)theConst4;
	const float theConst5f = (float)theConst5;

    float v0, v1, v2, v3, v4, v5;
    int *hit;

	// We used to do a lot of logic here for dynamic precision, but we have greatly simplified it
	// to help with compatibility with the Gpu checker.
	maxValue = std::max(
		{
			1000.0f, // largest numerical coefficient
			(
				1000.0f * std::abs(theConst5f) +
				500.0f * std::abs(theConst4f) +
				100.0f * std::abs(theConst3f) +
				60.0f * std::abs(theConst2f) +
				30.0f * std::abs(theConstf) +
				15.0f
			) // largest possible sum
		});

	floatTol = getFloatPrecisionBasedOnMaxValue(maxValue);
	doubleTol = getDoublePrecisionBasedOnMaxValue(maxValue);

    std::vector<int*> *hits = new std::vector<int*>();

    // note that these loops use <= (less than or EQUAL TO)
    for (int u = loopStartEnds[0]; u <= loopStartEnds[1]; u++) {
        v5 = ((u < 0) ? -LUT[-u] : LUT[u]) * theConst5f;

        for (int v = loopStartEnds[2]; v <= loopStartEnds[3]; v++) {
            v4 = v5 + ((v < 0) ? -LUT[-v] : LUT[v]) * theConst4f;
            
            for (int w = loopStartEnds[4]; w <= loopStartEnds[5]; w++) {
				v3 = v4 + ((w < 0) ? -LUT[-w] : LUT[w]) * theConst3f;
                
                for (int x = loopStartEnds[6]; x <= loopStartEnds[7]; x++) {
				    v2 = v3 + ((x < 0) ? -LUT[-x] : LUT[x]) * theConst2f;

                    for (int y = loopStartEnds[8]; y <= loopStartEnds[9]; y++) {
                        v1 = v2 + ((y < 0) ? -LUT[-y] : LUT[y]) * theConstf;

                        for (int z = loopStartEnds[10]; z <= loopStartEnds[11]; z++) {
                            v0 = v1 + ((z < 0) ? -LUT[-z] : LUT[z]);

                            if (FLOAT_BASICALLY_EQUAL(v0, needlef, floatTol)) {
								// TODO: Increment counter of float hits
                                floatHitCount++;
								//printf("floatHitCount=%ld\n", floatHitCount);
								//printf("double first two here is %10.10lf and %10.10lf\n", doubleLUT[u], doubleLUT[v]);
                                //printHit(LUT.data(), u,v,w,x,y,z);

								// since our float was in range, calculate the double value and check for a "real hit"
								doubleValue = ((u < 0) ? -doubleLUT[-u] : doubleLUT[u]) * theConst5
									+ ((v < 0) ? -doubleLUT[-v] : doubleLUT[v]) * theConst4
									+ ((w < 0) ? -doubleLUT[-w] : doubleLUT[w]) * theConst3
									+ ((x < 0) ? -doubleLUT[-x] : doubleLUT[x]) * theConst2
									+ ((y < 0) ? -doubleLUT[-y] : doubleLUT[y]) * (double)theConst
									+ ((z < 0) ? -doubleLUT[-z] : doubleLUT[z]);
								if (DOUBLE_BASICALLY_EQUAL(doubleValue, needle, doubleTol)) {
									// TODO: These are the real hits!
									hit = new int[6] {u, v, w, x, y, z};
                                	hits->push_back(hit);
									printf("Real hit!\n");
								}
                            }
                        }
                    }
                }
            }
        }
    }

	// Leaving this printf in to help prevent floatHitCount from being optimized out
	//printf("CpuQuinticFirstChecker: floatHitCount ending as %ld\n", floatHitCount);
    return hits;
}