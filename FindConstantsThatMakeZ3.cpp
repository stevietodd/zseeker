#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <ctime>
#include <numeric>
#include <limits>
#include <complex>
#include <mysql/mysql.h>
#include <cstring>
#include <cstdlib>
using namespace std;


#include <array>
#include <numeric>
#include <vector>

constexpr long double ld(int i, int j)
{
	return ((long double)i / j);
}

inline constexpr auto LUT = []
{
    constexpr auto LUT_LoopSize = 1000;
    //std::array<ResultT, (LUT_LoopSize*LUT_LoopSize*2)> arr = {};
	//ResultT arr[1'216'773] {0};					//TODO THIS IS WHERE I LEFT OFF 6/6/23
	//std::array<float, (1'216'773)> arr = {};
	std::array<long double, (608'384)> arr = {};

	// 8/21/25 I am refactoring the LUT to no longer have the 6 cutoffs at the front nor have negative values.
	
	// I'm having trouble figuring out how to save off the cutoff15, cutoff30, etc.
	// variables separately so I'm hacking this up and making the resulting lookup table
	// start with the 6 cutoff values I'm looking for. Hence why pos starts at 6. In other words:
	// LUT[0] = cutoff15 = 143 now, was 149 (so 150) only positives, all 292 (so array size needs to be 293)
	// LUT[1] = cutoff30 = 555 now, was 561 (so 562) only positives, all 1,116 (so 1,117)
	// LUT[2] = cutoff60 = 2,203 now, was 2,209 (so 2,210) only positives, all 4,412 (so 4,413)
	// LUT[3] = cutoff100 = 6,087 now, was 6,093 (so 6,094) only positives, all 12,180 (so 12,181)
	// LUT[4] = cutoff500 = 152.231 now, was 152,237 (so 152,238) only positives, all 304,468 (so 304,469)
	// LUT[5] = cutoff1000 = actualSize = 608,383 now, was 608,389 (so 608,390) only positives, all 1,216,772 (so 1,216,773)
	// LUT[6] = 0;
	// LUT[7] = 1;
	// LUT[8] = -1;
	// ...and so on...
	int pos = 0;

	// store zero separately
	arr[pos++] = ld(0,1);

	// store 1 and -1 separately
	arr[pos++] = ld(1,1);
	//arr[pos++] = f(-1,1);

    for (int i = 2; i <= LUT_LoopSize; i++)
    {
		for (int j = 1; j < i; j++)
		{
			if (std::gcd(i,j) > 1) {
				continue;
			}

			arr[pos++] = ld(j,i);
			//arr[pos++] = f(-j,i);
			arr[pos++] = ld(i,j);
			//arr[pos++] = f(-i,j);
		}
        
		// note we subtract 1 from pos because we've already incremented it
		// if (i == 15) {
		// 	arr[0] = pos - 1; // cutoff15
		// } else if (i == 30) {
		// 	arr[1] = pos - 1; // cutoff30
		// } else if (i == 60) {
		// 	arr[2] = pos - 1; // cutoff60
		// } else if (i == 100) {
		// 	arr[3] = pos - 1; // cutoff100
		// } else if (i == 500) {
		// 	arr[4] = pos - 1; // cutoff500
		// } else if (i == 1000) {
		// 	arr[5] = pos - 1; // cutoff1000
		// }
			
    }

    return arr;
}();

char* getCurrentTimeString() {
	std::time_t currTime = std::time(nullptr);
	return std::asctime(std::localtime(&currTime));
}

// MySQL connection configuration
struct MySQLConfig {
    const char* host;
    const char* user;
    const char* password;
    const char* database = "z3research";
    unsigned int port = 3306;
    
    // Constructor that reads from environment variables
    MySQLConfig() {
        host = getenv("DB_HOST");
        user = getenv("DB_USER");
        password = getenv("DB_PASSWORD");
        
        // Validate that required environment variables are set
        if (!host) {
            cerr << "Error: DB_HOST environment variable is not set" << endl;
            exit(1);
        }
        if (!user) {
            cerr << "Error: DB_USER environment variable is not set" << endl;
            exit(1);
        }
        if (!password) {
            cerr << "Error: DB_PASSWORD environment variable is not set" << endl;
            exit(1);
        }
    }
};

// Initialize MySQL connection
MYSQL* initializeMySQLConnection(const MySQLConfig& config) {
    MYSQL* mysql = mysql_init(nullptr);
    if (!mysql) {
        cerr << "Error: mysql_init failed" << endl;
        return nullptr;
    }
    
    if (!mysql_real_connect(mysql, config.host, config.user, config.password, 
                           config.database, config.port, nullptr, 0)) {
        cerr << "Error: mysql_real_connect failed: " << mysql_error(mysql) << endl;
        mysql_close(mysql);
        return nullptr;
    }
    
    return mysql;
}

// Structure to hold a single result for batch insertion
struct ResultRow {
    int w, x, y, z;
    long double root1, root2, root3;
};

// Insert multiple results in a batch
bool batchInsertResults(MYSQL* mysql, const std::vector<ResultRow>& results) {
    if (results.empty()) {
        return true;
    }
    
    // Start building the batch INSERT statement
    std::string query = "INSERT INTO z3_cubic_roots (mn_lut_index, pq_lut_index, rs_lut_index, zroot1, zroot2, zroot3) VALUES ";
    
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        
        // Format each row
        char rowBuffer[512];
        snprintf(rowBuffer, sizeof(rowBuffer),
            "(%d, %d, %d, %.20Lf, %.20Lf, %.20Lf)",
            result.x, result.y, result.z,
            result.root1, result.root2, result.root3);
        
        query += rowBuffer;
        
        // Add comma between rows (except for the last one)
        if (i < results.size() - 1) {
            query += ", ";
        }
    }
    
    // Execute the batch insert
    if (mysql_query(mysql, query.c_str())) {
        cerr << "Error: batch mysql_query failed: " << mysql_error(mysql) << endl;
        return false;
    }
    
    cout << "Successfully inserted " << results.size() << " results in batch." << endl;
    return true;
}

long double cubicroot_vanderbilt(long double a, long double b, long double c, long double d)
{
	// find one real root of ax^3 + bx^2 + cx + d
	long double p = b / (a * (long double)-3);
	complex<long double> q(powl(p, (long double)3) + ((b*c - a*d*(long double)3) / (powl(a,2)*(long double)6)));
	long double r = c / (a * (long double)3);
	complex<long double> s(powl(real(q),(long double)2) + powl(r-powl(p,(long double)2),(long double)3), 0);
	s = sqrt(s);

	return (real(pow(q+s, ((long double)1/3)) + pow(q-s, ((long double)1/3))) + p);
}

void cubicroot_wolfram(long double a, long double b, long double c, long double d, long double *resultArray)
{
	// find real roots of ax^3 + bx^2 + cx + d and return as [x0, x1, x2] in resultArray
	long double a2 = b / a;
	long double a1 = c / a;
	long double a0 = d / a;
	long double bigq = (3 * a1 - powl(a2,(long double)2)) / (long double)9;
	long double bigr = ((long double)9 * a2 * a1 - (long double)27 * a0 - (long double)2 * powl(a2, (long double)3)) / (long double)54;
	long double bigd = powl(bigq, (long double)3) + powl(bigr, (long double)2);

	//cout << "D=" << bigd << endl;

	if (bigq == 0 && bigr == 0 && bigd == 0) {
		// three identical real roots
		resultArray[0] = cbrtl(a0);
		resultArray[1] = 0;
		resultArray[2] = 0;
	} else if (bigd <= 0) {
		// all real roots, possibly unique
		long double theta = acos(bigr / sqrt((-1)*powl(bigq, (long double)3)));
		resultArray[0] = 2 * sqrt((-1) * bigq) * cos(theta / 3) - a2 / 3;
		resultArray[1] = 2 * sqrt((-1) * bigq) * cos((theta + 2*M_PIl) / 3) - a2 / 3;
		resultArray[2] = 2 * sqrt((-1) * bigq) * cos((theta + 4*M_PIl) / 3) - a2 / 3;
	} else { // bigd > 0
		// one real root, two complex conjugate roots
		resultArray[0] = cbrt(bigr + sqrt(bigd))
			+ cbrt(bigr - sqrtl(bigd))
			- (a2 / 3);
		resultArray[1] = 0;
		resultArray[2] = 0;
	}
}

void quadraticroot_wolfram(long double a, long double b, long double c, long double *resultArray)
{
	// find real roots of ax^2 + bx + cx return as [x0, x1]
	long double bigd = powl(b, (long double)2) - (4 * a * c);

	//cout << "D=" << bigd << endl;

	if (bigd < 0) {
		// two complex conjugate roots which I don't care about
		resultArray[0] = 0;
		resultArray[1] = 0;
	} else if (bigd == 0) {
		// two identical real roots
		resultArray[0] = (-1) * b / (2 * a);
		resultArray[1] = 0;
	} else { // bigd > 0
		// two unique real roots
		resultArray[0] = ((-1) * b + sqrt(bigd)) / (2 * a);
		resultArray[1] = ((-1) * b - sqrt(bigd)) / (2 * a);
	}
}

long double cubicroot_quoracopytweak(long double a, long double b, long double c, long double d)
{

long double e,f,g,h,i,j,k,l,m,n,p,r,s,t,u,x1,x2,x3;

int w;

//e=2.7182818284590;
e = M_El;

f=((3*c/a)-(b*b/(a*a)))/3; // ** bracketed (a*a)!

g=((2*b*b*b/(a*a*a))-(9*b*c/(a*a))+(27*d/a))/27; // ** brackets!

h=(g*g/4)+(f*f*f/27);

i=sqrt(((g*g/4)-h));

j=cbrtl(i);

p=(b/(3*a))*(-1);

r=(-1)*(g/2)+sqrt(h);

s=cbrtl(r); //exp(log10(r)/log10(e)/3);

t=(-1)*(g/2)-sqrt(h);

u=cbrtl(t);

if (h>0) w=1;

if (h<=0) w=3;

if ((f==0) && (g==0) && (h==0)) w=2;

switch (w){

case 1:

x1=(s+u)-(b/(3*a));

x2=(-1)*(s+u)/2-(b/(3*a));

x3=(s-u)*sqrt((long double)3)/2;

printf("\nA 3 pont:\n%Lf\n%Lf +i*%Lf\n%Lf -i*%Lf", x1, x2, x3, x2, x3);

break;

case 2:

x1=cbrtl(d/a)*(-1); //exp(log10(d/a)/log10(e)/3)*(-1);

printf("\n There is a line:\n%Lf", x1);

break;

case 3:
// future stevie: just adjust this section to use wolfram and it should work
k=acos((-1)*(g/(2*i)));

l=j*(-1);

m=cos(k/3);

n=sqrt((long double)3)*sin(k/3);

x1=2*j*cos(k/3)-(b/(3*a));

x2=l*(m+n)+p;

x3=l*(m-n)+p;

printf("\nA 3 Roots:\n%Lf\n%Lf\n%Lf", x1, x2, x3);

break;

}

}

int main(int argc, char *argv[])
{
	int kstart = -10000;
	int uplim3 = 100;
	int uplim2 = 60;
	int uplim1 = 30;
	int uplim = 15;
	int lowlim3 = -1 * uplim3;
	int lowlim2 = -1 * uplim2;
	int lowlim1 = -1 * uplim1;
	int lowlim = -1 * uplim;
	long double jk, mn, pq, rs;
	long double val, v1, v2, v3, absDif;
	long double z3 = riemann_zetal((long double)3);
	long double *roots = new long double[3];

	typedef std::numeric_limits< long double > ldbl;

	cout.precision(ldbl::max_digits10);

	// long double *yo = quadraticroot_wolfram(4,1,4);
	// cout << "x1=" << yo[0] << endl;
	// cout << "x2=" << yo[1] << endl;
	// delete yo;
	// return 0;

	// long double *result = cubicroot_wolfram(0, 0, 1, 1);
	// cout << "x1=" << result[0] << endl;
	// cout << "x2=" << result[1] << endl;
	// cout << "x3=" << result[2] << endl;
	// delete result;
	// return 0;

	// Initialize MySQL connection
	MySQLConfig config; // This will read from environment variables and validate them
	MYSQL* mysql = initializeMySQLConnection(config);
	if (!mysql) {
		cerr << "Failed to initialize MySQL connection. Exiting." << endl;
		return 1;
	}
	
	cout << "Connected to MySQL database successfully." << endl;

	// Updated loop boundaries to go from negative to positive ranges instead of starting from 6
	// NOTE: We don't use the first 4 elements because we're not dealing with quintics or quartics here, but this makes
	// us consistent with other classes
	int loopStartEnds[12] = {-608'383, 608'383, -152'231, 152'231, -6'087, 6'087, -2'203, 2'203, -555, 555, -143, 143};
	int wStart = loopStartEnds[4];
	int wEnd = loopStartEnds[5];

	switch (argc) {
		case 2:
		{
			std::istringstream ss2(argv[1]);
			if (!(ss2 >> wStart)) {
				cerr << "Invalid value for wstart: " << argv[1] << endl;
				// if there was a problem, use the default values
				wStart = loopStartEnds[4];
				wEnd = loopStartEnds[5];
			} else {
				// since they specified a value, just run the one loop
				wEnd = wStart;
			}
		}
	}

	 //TODO: Use degree for way more things than just processing loopRanges
	 // if loopRanges is non-null, find first level with positive values (-1 indicates use default) and use those
	 // (WRONG) note that we ignore any level after that since we don't want to skip coeffs in later loops (WRONG)
	 // note that we DO allow all levels to be updated now but warn the user that they may have an incomplete search
	//  if (loopRanges != NULL) {
	// 	 // loopRanges must have (2*(degree+1)) elements. Format is [zStart, zEnd, yStart, yEnd, ...]
	// 	 for (int loopRangeInd = 0; loopRangeInd < (2*(degree+1)); loopRangeInd++) {
	// 		 //TODO: Make this not so hacky and stupid
	// 		 if (loopRanges->at(loopRangeInd) < USE_DEFAULT) {
	// 			 // they are setting a non-default value, so update loopStartEnds
	// 			 loopStartEnds[loopRangeInd] = loopRanges->at(loopRangeInd);
	// 			 std::cout << "WARNING: You have set a non-standard loop range. Your search may be incomplete" << std::endl;
	// 		 }
	// 	 }
	//  }

	for (int w = wStart; w <= wEnd; w++) {
		jk = ((w < 0) ? -LUT[-w] : LUT[w]);
		cout << "w=" << w << ", " << getCurrentTimeString();

		for (int x = loopStartEnds[6]; x <= loopStartEnds[7]; x++) {
			mn = ((x < 0) ? -LUT[-x] : LUT[x]);
			
			// Vector to collect results for this x iteration
			std::vector<ResultRow> xResults;
			
			for (int y = loopStartEnds[8]; y <= loopStartEnds[9]; y++) {
				pq = ((y < 0) ? -LUT[-y] : LUT[y]);
				for (int z = loopStartEnds[10]; z <= loopStartEnds[11]; z++) {
					rs = ((z < 0) ? -LUT[-z] : LUT[z]);
					// handle quadratic, linear, constant logic
					if (w == 0) {
						// don't do cubic formula
						if (x == 0) {
							// don't even do quadratic formula
							if (y == 0) {
								// do nothing at all because rs = z3 is not gonna happen
								// TODO: should this be a break?
								break;
							} else {
								// (p / q)c + (r / s) = z3
								roots[0] = (z3 - (rs / pq));
								roots[1] = 0;  // make sure second element is reset
								roots[2] = 0;  // make sure third element is reset
							}
						} else {
							// (m / n)c^2 + (p / q)c + (r / s) = z3
							quadraticroot_wolfram(
								mn,
								pq,
								rs - z3,
								roots
							);
							roots[2] = 0;  // make sure third element is reset
						}
					} else {
						// (j / k)c^3 + (m / n)c^2 + (p / q)c + (r / s) = z3
						cubicroot_wolfram(
							jk,
							mn,
							pq,
							rs - z3,
							roots
						);
					}

					// Add result to batch collection instead of individual insert
					ResultRow result;
					result.w = w;
					result.x = x;
					result.y = y;
					result.z = z;
					result.root1 = roots[0];
					result.root2 = roots[1];
					result.root3 = roots[2];
					xResults.push_back(result);
				}
			}
			
			// Batch insert all results collected for this x iteration
			if (!xResults.empty()) {
				if (!batchInsertResults(mysql, xResults)) {
					cerr << "Failed to batch insert results for x=" << x << ". Continuing..." << endl;
				}
				xResults.clear(); // Clear the vector for the next x iteration
			}
		}
	}
	
	// Clean up MySQL connection
	mysql_close(mysql);
	cout << "MySQL connection closed." << endl;
	
	// Clean up memory
	delete[] roots;
	
	return 0;
}

int oldMain(int argc, char *argv[])
{
	int jstart = 0;
	int kstart = -10000;
	int uplim3 = 100;
	int uplim2 = 60;
	int uplim1 = 30;
	int uplim = 15;
	int lowlim3 = -1 * uplim3;
	int lowlim2 = -1 * uplim2;
	int lowlim1 = -1 * uplim1;
	int lowlim = -1 * uplim;
	long double val, v1, v2, v3, absDif;
	long double z3 = riemann_zetal((long double)3);
	long double *roots = new long double[3];

	typedef std::numeric_limits< long double > ldbl;

	cout.precision(ldbl::max_digits10);

	// long double *yo = quadraticroot_wolfram(4,1,4);
	// cout << "x1=" << yo[0] << endl;
	// cout << "x2=" << yo[1] << endl;
	// delete yo;
	// return 0;

	// long double *result = cubicroot_wolfram(0, 0, 1, 1);
	// cout << "x1=" << result[0] << endl;
	// cout << "x2=" << result[1] << endl;
	// cout << "x3=" << result[2] << endl;
	// delete result;
	// return 0;

	ofstream csvFile;
	char csvFileName[25];
	csvFile.precision(ldbl::max_digits10);

	switch (argc) {
		case 3:
		{
			std::istringstream ss(argv[2]);
			if (!(ss >> kstart)) {
				cerr << "Invalid value for kstart: " << argv[2] << endl;
				kstart = -10000;
			}
		}
		case 2:
		{
			std::istringstream ss2(argv[1]);
			if (!(ss2 >> jstart)) {
				cerr << "Invalid value for jstart: " << argv[1] << endl;
				jstart = 1;
			}
		}
	}

	for (int j = jstart; j <= uplim3; j++) {
    	cout << "j=" << j << ", " << getCurrentTimeString();
    	sprintf (csvFileName, "z3roots_j%d.csv", j);
		csvFile.open(csvFileName);

		// allow for setting where k starts on the first loop only
		if (j != jstart || kstart == -10000) {
			kstart = lowlim3;
		}

		for (int k = kstart; k <= uplim3; k++) {
			if (k % 10 == 0) {
				cout << "k=" << k << ", " << getCurrentTimeString();
			}
			// skip if we would have already done this work
			if (k==0 || (j != 0 && gcd(j,k) > 1)) {
				continue;
			}

			for (int m = 0; m <= uplim2; m++) {
				//cout << "m=" << m << ", " << getCurrentTimeString() << endl;
				for (int n = lowlim2; n <= uplim2; n++) {
					// skip if we would have already done this work
					if (n==0 || (m != 0 && gcd(m,n) > 1)) {
						continue;
					}
          
					for (int p = 0; p <= uplim1; p++) {
						for (int q = lowlim1; q <= uplim1; q++) {
							// skip if we would have already done this work
							if (q==0 || (p != 0 && gcd(p,q) > 1)) {
								continue;
							}
							
							for (int r = 0; r <= uplim; r++) {
								for (int s = lowlim; s <= uplim; s++) {
									// skip if we would have already done this work
									if (s==0 || (r != 0 && gcd(r,s) > 1)) {
										continue;
									}

									// if (k == -65 && m == 6 && n == 13 && p == 5 && q == 11 && r == 1 && s == 10) {
									// 	cout << v3 << ":" << v2 << ":" << v1 << endl;
									// }
									
									// handle quadratic, linear, constant logic
									if (j == 0) {
										// don't do cubic formula
										if (m == 0) {
											// don't even do quadratic formula
											if (p == 0) {
												// do nothing at all because (r / s) = z3 is not gonna happen
												// TODO: should this be a break?
												break;
											} else {
												// (p / q)x + (r / s) = z3
												roots[0] = (z3 - ((long double)r / s)) * ((long double)q / p);
												roots[1] = 0;  // make sure second element is reset
												roots[2] = 0;  // make sure third element is reset
											}
										} else {
											// (m / n)x^2 + (p / q)x + (r / s) = z3
											quadraticroot_wolfram(
												(long double)m / n,
												(long double)p / q,
												((long double)r / s) - z3,
												roots
											);
											roots[2] = 0;  // make sure third element is reset
										}
									} else {
										// (j / k)x^3 + (m / n)x^2 + (p / q)x + (r / s) = z3
										cubicroot_wolfram(
											(long double)j / k,
											(long double)m / n,
											(long double)p / q,
											((long double)r / s) - z3,
											roots
										);
									}

									csvFile << roots[0] << "," << roots[1] << "," << roots[2] << "," <<
										j << "," << k << "," << m << "," << n << "," <<
										p << "," << q << "," << r << "," << s << "\n";

									if (r == 0) {
										// only evaluate zero loop once
										break;
									}
								}
							}

							if (p == 0) {
								// only evaluate zero loop once
								break;
							}
						}
					}

					if (m == 0) {
						// only evaluate zero loop once
						break;
					}
				}
				
				// flush our file just to be safe
				csvFile << flush;
			}

			if (j == 0) {
				// only evaluate zero loop once
				break;
			}
		}

		// close the file
		csvFile << flush;
		csvFile.close();
	}
}