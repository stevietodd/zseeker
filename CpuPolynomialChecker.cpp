#include "CpuPolynomialChecker.hpp"
#include "lookupTable.hpp"
#include "math.hpp"
#include <ctime> // can remove if getCurrentTimeString is removed
#include <iostream> // can remove if not using cout

char* getCurrentTimeString() {
	std::time_t currTime = std::time(nullptr);
	return std::asctime(std::localtime(&currTime));
}

std::vector<float*>* CpuPolynomialChecker::findHits(
            const float needle,
            const float theConst,
            const int degree,
            const std::vector<float> *coeffArray,
            const std::vector<int> *loopRanges
)
{
    // TODO: This sucks. Change this
    // note that even elements are LUT[0] through LUT[5]
    int loopStartEnds[12] = {6, 292, 6, 1'116, 6, 4'412, 6, 12'180, 6, 304'468, 6, 1'216'772};

    //TODO: Use degree for way more things than just processing loopRanges
    // if loopRanges is non-null, find first level with positive values (-1 indicates use default) and use those
    // note that we ignore any level after that since we don't want to skip coeffs in later loops
    if (loopRanges != NULL) {
        // loopRanges must have (2*(degree+1)) elements. Format is [zStart, zEnd, yStart, yEnd, ...]
        for (int loopRangeInd = 0; loopRangeInd < (2*(degree+1)); loopRangeInd++) {
            //TODO: Make this not so hacky and stupid
            if (loopRanges->at(loopRangeInd) >= 0) {
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

    float v0, v1, v2, v3, v4, v5, *hit;

    std::vector<float*> *hits = new std::vector<float*>();

    // note that these loops use <= (less than or EQUAL TO)
    for (int z = loopStartEnds[0]; z <= loopStartEnds[1]; z++) {
		v0 = LUT[z];
	
		for (int y = loopStartEnds[2]; y <= loopStartEnds[3]; y++) {
			v1 = v0 + LUT[y] * theConst;

			for (int x = loopStartEnds[4]; x <= loopStartEnds[5]; x++) {
				v2 = v1 + LUT[x] * theConst2;

				for (int w = loopStartEnds[6]; w <= loopStartEnds[7]; w++) {
					v3 = v2 + LUT[w] * theConst3;
                    //std::cout << "w=" << w << ", " << getCurrentTimeString();

                    for (int v = loopStartEnds[8]; v <= loopStartEnds[9]; v++) { // took 46-48 minutes per w-loop but gets results
                        v4 = v3 + LUT[v] * theConst4;

                        for (int u = loopStartEnds[10]; u <= loopStartEnds[11]; u++) {
                            v5 = v4 + LUT[u] * theConst5;

                            if (FLOAT_BASICALLY_EQUAL(v5, ZETA5)) {
                                hit = new float[6] {LUT[u], LUT[v], LUT[w], LUT[x], LUT[y], LUT[z]};
                                hits->push_back(hit);
                            }
                        }
                    }
				}
			}
		}
	}

    return hits;
}