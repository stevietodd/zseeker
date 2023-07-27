#include <iostream>
#include <sstream>
#include <cmath>
#include <ctime>
#include <numeric>
#include <limits>
#include "cudawrapper.hpp"
using namespace std;

char* getCurrentTimeString() {
	std::time_t currTime = std::time(nullptr);
	return std::asctime(std::localtime(&currTime));
}

#include <array>
#include <vector>
//using ResultT = float;
constexpr float f(int i, int j)
{
    return ((float)i / j);
}

constexpr auto LUT = []
{
    constexpr auto LUT_LoopSize = 1000;
    //std::array<ResultT, (LUT_LoopSize*LUT_LoopSize*2)> arr = {};
	//ResultT arr[1'216'773] {0};					//TODO THIS IS WHERE I LEFT OFF 6/6/23
	std::array<float, (1'216'773)> arr = {};
	
	// I'm having trouble figuring out how to save off the cutoff15, cutoff30, etc.
	// variables separately so I'm hacking this up and making the resulting lookup table
	// start with the 6 cutoff values I'm looking for. Hence why pos starts at 6. In other words:
	// LUT[0] = cutoff15 = 292 (so array size needs to be 293)
	// LUT[1] = cutoff30 = 1,116 (so 1,117)
	// LUT[2] = cutoff60 = 4,412 (so 4,413)
	// LUT[3] = cutoff100 = 12,180 (so 12,181)
	// LUT[4] = cutoff500 = 304,468 (so 304,469)
	// LUT[5] = cutoff1000 = actualSize = 1,216,772 (so 1,216,773)
	// LUT[6] = 0;
	// LUT[7] = 1;
	// LUT[8] = -1;
	// ...and so on...
	int pos = 6;

	// store zero separately
	arr[pos++] = f(0,1);

	// store 1 and -1 separately
	arr[pos++] = f(1,1);
	arr[pos++] = f(-1,1);

    for (int i = 2; i <= LUT_LoopSize; i++)
    {
		for (int j = 1; j < i; j++)
		{
			if (gcd(i,j) > 1) {
				continue;
			}

			arr[pos++] = f(j,i);
			arr[pos++] = f(-j,i);
			arr[pos++] = f(i,j);
			arr[pos++] = f(-i,j);
		}
        
		// note we subtract 1 from pos because we've already incremented it
		if (i == 15) {
			arr[0] = pos - 1; // cutoff15
		} else if (i == 30) {
			arr[1] = pos - 1; // cutoff30
		} else if (i == 60) {
			arr[2] = pos - 1; // cutoff60
		} else if (i == 100) {
			arr[3] = pos - 1; // cutoff100
		} else if (i == 500) {
			arr[4] = pos - 1; // cutoff500
		} else if (i == 1000) {
			arr[5] = pos - 1; // cutoff1000
		}
			
    }

    return arr;
}();

void printHit(int i5, int i4, int i3, int i2, int i1, int i0)
{
	cout << "(" << i5 << "," << i4 << "," << i3 << "," << i2 << "," << i1 << "," << i0 << "): " <<
		LUT[i5] << "c^5 + " << LUT[i4] << "c^4 + " << LUT[i3] << "c^3 + " << LUT[i2] << "c^2 + " <<
		LUT[i1] << "c + " << LUT[i0] << " = HIT!\n";
}

int main(int argc, char *argv[])
{
	int estart = 0;
	int fstart = -10000;
	const float lnpi = M_PI;  //TODO: Note this is actually pi now, not ln(pi)!
	int prevlim = -1;
	int uplim5 = 1000;
	int uplim4 = 500;
	int uplim3 = 100;
	int uplim2 = 60;
	int uplim1 = 30;
	int uplim = 15;
	int lowlim5 = -1 * uplim5;
	int lowlim4 = -1 * uplim4;
	int lowlim3 = -1 * uplim3;
	int lowlim2 = -1 * uplim2;
	int lowlim1 = -1 * uplim1;
	int lowlim = -1 * uplim;
	const float z5 = 1.036927755143369926331365486457034168L; //riemann_zetal((long double)5);
	const float lnpi2 = powl(lnpi, (float)2);
	const float lnpi3 = powl(lnpi, (float)3);
	const float lnpi4 = powl(lnpi, (float)4);
	const float lnpi5 = powl(lnpi, (float)5);
	const float v0max = (float)uplim;
	const float v1max = (float)uplim1 * lnpi + v0max;
	const float v2max = (float)uplim2 * lnpi2 + v1max;
	const float v3max = (float)uplim3 * lnpi3 + v2max;
	const float v4max = (float)uplim4 * lnpi4 + v3max;
	const float v1BreakoutHigh = z5 + v0max;
	const float v1BreakoutLow = z5 - v0max;
	const float v2BreakoutHigh = z5 + v1max;
	const float v2BreakoutLow = z5 - v1max;
	const float v3BreakoutHigh = z5 + v2max;
	const float v3BreakoutLow = z5 - v2max;
	const float v4BreakoutHigh = z5 + v3max;
	const float v4BreakoutLow = z5 - v3max;
	const float v5BreakoutHigh = z5 + v4max;
	const float v5BreakoutLow = z5 - v4max;

	vector<float> *output = new vector<float>();
	
	float v0, v1, v2, v3, v4, v5, valDif;
	const float *LUTptr = LUT.data();
	bool useGPU = true;

	typedef std::numeric_limits< float > ldbl;

	cout.precision(ldbl::max_digits10);

	cout << "Yo" << endl;
	cout << sizeof(LUT) << endl;
	cout << "cutoff15=" << LUT[0] << "\n";
	// cout << "cutoff30=" << LUT[1] << "\n";
	// cout << "cutoff60=" << LUT[2] << "\n";
	// cout << "cutoff100=" << LUT[3] << "\n";
	// cout << "cutoff500=" << LUT[4] << "\n";
	// cout << "cutoff1000=" << LUT[5] << "\n";
	// cout << "v0max=" << v0max << "\n";
	// cout << "v1max=" << v1max << "\n";
	// cout << "v2max=" << v2max << "\n";
	// cout << "v3max=" << v3max << "\n";
	// cout << "v4max=" << v4max << "\n";
	//cout << "LUTval=" << LUT[6] << "," << LUT[9] << "," << LUT[3492] << "," << LUT[3214] << "," << LUT[197] << "," << LUT[146] << "\n";
	//cout << "LUTval2=" << LUT[6] << "," << LUT[30] << "," << LUT[7465] << "," << LUT[3695] << "," << LUT[754] << "," << LUT[241] << "\n";

	switch (argc) {
		case 2:
		{
			useGPU = false;
		}
	}

	for (int z = 6; z <= 292; z++) { // LUT[0]
		v0 = LUT[z];
	
		for (int y = 6; y <= 1116; y++) { // LUT[1]
			v1 = v0 + LUT[y] * lnpi;

			for (int x = 6; x <= 4412; x++) { // LUT[2]
				v2 = v1 + LUT[x] * lnpi2;

				for (int w = 6; w <= 12180; w++) { // LUT[3]
					v3 = v2 + LUT[w] * lnpi3;

							// if (w % 100 == 0) {
					cout << "w=" << w << ", " << getCurrentTimeString();
					if (useGPU) {
						output = testForZeta5OnGPU(lnpi, v3, LUTptr, 304468, 1216772);  // took 12-13 minutes per w-loop but supposedly isn't getting right results
					} else {
						for (int v = 6; v <= 304468; v++) { // LUT[4]   // took 46-48 minutes per w-loop but gets results
							v4 = v3 + LUT[v] * lnpi4;

							//cout << "v=" << v << ", " << getCurrentTimeString();
							for (int u = 6; u <= 1216772; u++) { // LUT[5]
								v5 = v4 + LUT[u] * lnpi5;
								//cout << "u=" << u << ", " << getCurrentTimeString();
								valDif = v5 - z5;
								if (valDif < .0000001 && valDif > -.0000001) {
									printHit(u,v,w,x,y,z); 
								}
							}
						}
					}
				}
			}
		}
	}

	// for (int u = 6; u <= 1216772; u++) { // LUT[5]
	// 	cout << "u=" << u << ", " << getCurrentTimeString();
	// /*for (int e = estart; e <= uplim5; e++) {
    // 	cout << "e=" << e << ", " << getCurrentTimeString();

	// 	// allow for setting where f starts on the first loop only
	// 	if (e != estart || fstart == -10000) {
	// 		fstart = lowlim5;
	// 	}

	// 	for (int f = fstart; f <= uplim5; f++) {
	// 		if (f % 10 == 0) {
	// 			cout << "f=" << f << ", " << getCurrentTimeString();
	// 		}
	// 		// skip if we would have already done this work
	// 		if (f==0 || (e != 0 && gcd(e,f) > 1)) {
	// 			continue;
	// 		}*/

	// 		v5 = LUT[u] * lnpi5;
	// 		//v5 = ((long double)e / f) * powl(lnpi, (long double)5);
	// 		if (v5 < v5BreakoutLow || v5 > v5BreakoutHigh) {
	// 			// we can't possibly get back to z5, so bust out
	// 			continue;
	// 		}

	// 		for (int v = 6; v <= 304468; v++) { // LUT[4]
	// 			cout << "v=" << v << ", " << getCurrentTimeString();
	// 		/*for (int g = 0; g <= uplim4; g++) {
	// 			cout << "g=" << g << ", " << getCurrentTimeString();
				
	// 			for (int h = lowlim4; h <= uplim4; h++) {
	// 				cout << "h=" << h << ", " << getCurrentTimeString();
					
	// 				// skip if we would have already done this work
	// 				if (h==0 || (g != 0 && gcd(g,h) > 1)) {
	// 					continue;
	// 				}*/

	// 				v4 = v5 + LUT[v] * lnpi4;
	// 				//v4 = v5 + ((long double)g / h) * powl(lnpi, (long double)4);
	// 				//v4 = ((long double)g / h) * powl(lnpi, (long double)4);  // THIS WAS THE OLD INCORRECT LINE :facepalm:
	// 				if (v4 < v4BreakoutLow || v4 > v4BreakoutHigh) {
	// 					// we can't possibly get back to z5, so bust out
	// 					continue;
	// 				}

	// 				for (int w = 6; w <= 12180; w++) { // LUT[3]
	// 					if (w % 100 == 0) {
	// 						cout << "w=" << w << ", " << getCurrentTimeString();
	// 						//return 0; //TODO: REMOVE ME AFTER PROFILING!
	// 					}
	// 				/*for (int j = 0; j <= uplim3; j++) {
	// 					cout << "j=" << j << ", " << getCurrentTimeString();

	// 					for (int k = lowlim3; k <= uplim3; k++) {
	// 						cout << "k=" << k << ", " << getCurrentTimeString() << endl;

	// 						// skip if we would have already done this work
	// 						if (k==0 || (j != 0 && gcd(j,k) > 1)) {
	// 							continue;
	// 						}*/

	// 						v3 = v4 + LUT[w] * lnpi3;
	// 						//v3 = v4 + ((long double)j / k) * powl(lnpi, (long double)3);
	// 						//v3 = ((long double)j / k) * powl(lnpi, (long double)3);  // THIS WAS THE OLD INCORRECT LINE :facepalm:
	// 						if (v3 < v3BreakoutLow || v3 > v3BreakoutHigh) {
	// 							// we can't possibly get back to z5, so bust out
	// 							continue;
	// 						}

	// 						for (int x = 6; x <= 4412; x++) { // LUT[2]
	// 						/*for (int m = 0; m <= uplim2; m++) {
	// 							//cout << "m=" << m << ", " << getCurrentTimeString() << endl;
	// 							for (int n = lowlim2; n <= uplim2; n++) {
	// 								// skip if we would have already done this work
	// 								if (n==0 || (m != 0 && gcd(m,n) > 1)) {
	// 									continue;
	// 								}*/

	// 								v2 = v3 + LUT[x] * lnpi2;
	// 								//v2 = v3 + ((long double)m / n) * powl(lnpi, (long double)2);
	// 								if (v2 < v2BreakoutLow || v2 > v2BreakoutHigh) {
	// 									// we can't possibly get back to z5, so bust out
	// 									continue;
	// 								}
						
	// 								for (int y = 6; y <= 1116; y++) { // LUT[1]
	// 								/*for (int p = 0; p <= uplim1; p++) {
	// 									for (int q = lowlim1; q <= uplim1; q++) {
	// 										// skip if we would have already done this work
	// 										if (q==0 || (p != 0 && gcd(p,q) > 1)) {
	// 											continue;
	// 										}*/

	// 										v1 = v2 + LUT[y] * lnpi;
	// 										//v1 = v2 + ((long double)p / q) * lnpi;
	// 										if (v1 < v1BreakoutLow || v1 > v1BreakoutHigh) {
	// 											// we can't possibly get back to z5, so bust out
	// 											continue;
	// 										}
											
	// 										// look up table loop. note we start at 6 to avoid cutoff values
	// 										for (int z = 6; z <= 292; z++) // LUT[0]
	// 										{
	// 											val = v1 + LUT[z];
	// 											valDif = val - z5;
	// 											if (valDif < .000000000001L && valDif > -.000000000001L) {
	// 												printf("****(%d),(%d),(%d),(%d),(%d),(%d)****\n", u,v,w,x,y,z); 
	// 											}
	// 										}

	// 										/*for (int r = 0; r <= uplim; r++) {
	// 											for (int s = lowlim; s <= uplim; s++) {
	// 												// skip if we would have already done this work
	// 												if (s==0 || (r != 0 && gcd(r,s) > 1)) {
	// 													continue;
	// 												}

	// 												// if we already did this work in a previous run of the script
	// 												// with a lower uplim, move past it all
	// 												// if (j <= prevlim && abs(k) <= prevlim &&
	// 												// 	m <= prevlim && abs(n) <= prevlim &&
	// 												// 	p <= prevlim && abs(q) <= prevlim &&
	// 												// 	r <= prevlim && abs(s) <= prevlim) {
	// 												// 	s = min(uplim,prevlim);
	// 												// }

	// 												// if (k == -65 && m == 6 && n == 13 && p == 5 && q == 11 && r == 1 && s == 10) {
	// 												// 	cout << v3 << ":" << v2 << ":" << v1 << endl;
	// 												// }
													
	// 												val = v1 + ((long double)r / s);
	// 												absDif = abs(val - z5);
	// 												if (absDif < .000000000001) {
	// 													printf("****(%d/%d),(%d/%d),(%d/%d),(%d/%d),(%d/%d),(%d/%d)****\n", e,f,g,h,j,k,m,n,p,q,r,s); 
	// 												}

	// 												if (r == 0) {
	// 													// only evaluate zero loop once
	// 													break;
	// 												}
	// 											}
	// 										}*/

	// 										/*if (p == 0) {
	// 											// only evaluate zero loop once
	// 											break;
	// 										}
	// 									}*/
	// 								}

	// 								/*if (m == 0) {
	// 									// only evaluate zero loop once
	// 									break;
	// 								}
	// 							}*/
	// 						}

	// 						/*if (j == 0) {
	// 							// only evaluate zero loop once
	// 							break;
	// 						}
	// 					}*/
	// 				}

	// 				/*if (g == 0) {
	// 					// only evaluate zero loop once
	// 					break;
	// 				}
	// 			}*/
	// 		}

	// 		/*if (e == 0) {
	// 			// only evaluate zero loop once
	// 			break;
	// 		}
	// 	}*/
	// }

	return 0;
}