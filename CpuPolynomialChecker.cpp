#include "CpuPolynomialChecker.hpp"
#include "lookupTable.hpp"
#include "math.hpp"

// Provide implementation for the first method
std::vector<float>* CpuPolynomialChecker::findHits(const float theConst, const float needle, const std::vector<float> *coeffArray)
{
    const float theConst2 = powl(theConst, (float)2);
	const float theConst3 = powl(theConst, (float)3);
	const float theConst4 = powl(theConst, (float)4);
	const float theConst5 = powl(theConst, (float)5);

    float v0, v1, v2, v3, v4, v5;

    std::vector<float> *hits = new std::vector<float>();

    // TODO: stop hard-coding these values
    for (int z = 6; z <= 292; z++) { // LUT[0]
		v0 = LUT[z];
	
		for (int y = 6; y <= 1116; y++) { // LUT[1]
			v1 = v0 + LUT[y] * theConst;

			for (int x = 6; x <= 4412; x++) { // LUT[2]
				v2 = v1 + LUT[x] * theConst2;

				for (int w = 6; w <= 12180; w++) { // LUT[3]
					v3 = v2 + LUT[w] * theConst3;

                    for (int v = 6; v <= 304468; v++) { // LUT[4]   // took 46-48 minutes per w-loop but gets results
                        v4 = v3 + LUT[v] * theConst4;

                        for (int u = 6; u <= 1216772; u++) { // LUT[5]
                            v5 = v4 + LUT[u] * theConst5;
                            if (FLOAT_BASICALLY_EQUAL(v5, ZETA5)) {
                                //TODO: MASSIVE rewrite needed...hits should be array of coeffs, not one single float
                                hits->push_back(theConst);
                            }
                        }
                    }
				}
			}
		}
	}

    return hits;
}