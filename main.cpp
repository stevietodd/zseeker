#include <iostream>
#include <string>
#include <strings.h>
#include <mysql/mysql.h>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <vector>
#include "CpuPolynomialChecker.hpp"
#include "GpuPolynomialChecker.hpp"
#include "CubicRootSliceWorker.hpp"
#include "math.hpp"
#include "lookupTableAccessor.hpp"

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
            std::cerr << "Error: DB_HOST environment variable is not set" << std::endl;
            exit(1);
        }
        if (!user) {
            std::cerr << "Error: DB_USER environment variable is not set" << std::endl;
            exit(1);
        }
        if (!password) {
            std::cerr << "Error: DB_PASSWORD environment variable is not set" << std::endl;
            exit(1);
        }
    }
};

// Initialize MySQL connection
MYSQL* initializeMySQLConnection(const MySQLConfig& config) {
    MYSQL* mysql = mysql_init(nullptr);
    if (!mysql) {
        std::cerr << "Error: mysql_init failed" << std::endl;
        return nullptr;
    }
    
    if (!mysql_real_connect(mysql, config.host, config.user, config.password, 
                           config.database, config.port, nullptr, 0)) {
        std::cerr << "Error: mysql_real_connect failed: " << mysql_error(mysql) << std::endl;
        mysql_close(mysql);
        return nullptr;
    }
    
    return mysql;
}

// Fetch theConst from z3_cubic_roots table
double fetchTheConstFromDatabase(MYSQL* mysql) {
    const char* query = "SELECT zroot1, zroot2, zroot3 FROM z3_cubic_roots LIMIT 1";
    
    if (mysql_query(mysql, query)) {
        std::cerr << "Error: mysql_query failed: " << mysql_error(mysql) << std::endl;
        return M_PI; // fallback to PI if query fails
    }
    
    MYSQL_RES* result = mysql_store_result(mysql);
    if (!result) {
        std::cerr << "Error: mysql_store_result failed: " << mysql_error(mysql) << std::endl;
        return M_PI; // fallback to PI if no results
    }
    
    MYSQL_ROW row = mysql_fetch_row(result);
    if (!row || !row[0]) {
        std::cerr << "Error: No data found in z3_cubic_roots table" << std::endl;
        mysql_free_result(result);
        return M_PI; // fallback to PI if no data
    }
    
    double theConst = std::stod(row[0]);
    mysql_free_result(result);
    
    std::cout << "Fetched constant from database: " << theConst << std::endl;
    return theConst;
}


int main(int argc, char *argv[])
{
    PolynomialCheckerInterface *checker;
    std::vector<int*> *hits;
    long doubleHitCount = 0;
    double theConst;

    typedef std::numeric_limits< float > ldbl;
    std::cout.precision(ldbl::max_digits10);

    // Parse checker type from command line arguments
    switch (argc) {
        case 2:
        {
            if (strcasecmp(argv[1], "gl") == 0) {
                std::cout << "Using GpuQuinticLast" << std::endl;
                checker = new GpuQuinticLastChecker();
            } else if (strcasecmp(argv[1], "gf") == 0) {
                std::cout << "Using GpuQuinticFirst" << std::endl;
                checker = new GpuQuinticFirstChecker();
            } else if (strcasecmp(argv[1], "gfpo") == 0) {
                std::cout << "Using GpuQuinticFirstPositiveOnly" << std::endl;
                checker = new GpuQuinticFirstCheckerPositiveOnly();
            } else if (strcasecmp(argv[1], "cl") == 0) {
                std::cout << "Using CpuQuinticLast" << std::endl;
                checker = new CpuQuinticLastChecker();
            } else if (strcasecmp(argv[1], "cf") == 0) {
                std::cout << "Using CpuQuinticFirst" << std::endl;
                checker = new CpuQuinticFirstChecker();
            } else if (strcasecmp(argv[1], "cfwb") == 0) {
                std::cout << "Using CpuQuinticFirstWithBreakouts" << std::endl;
                checker = new CpuQuinticFirstWithBreakoutsChecker();
            } else if (strcasecmp(argv[1], "megaman") == 0) {
                std::cout << "Using Hack" << std::endl;
                checker = new GpuQuinticFirstChecker();
                // Create loopRanges: zStart=-5, zEnd=5, all others USE_DEFAULT
                // Format: [zStart, zEnd, yStart, yEnd, xStart, xEnd, cubicStart, cubicEnd, quartStart, quartEnd, quintStart, quintEnd]
                std::vector<int> loopRanges = {
                    -6, 6,  // zStart, zEnd
                    -6, 6,  // yStart, yEnd
                    USE_DEFAULT, USE_DEFAULT,  // xStart, xEnd
                    USE_DEFAULT, USE_DEFAULT,  // cubicStart, cubicEnd
                    USE_DEFAULT, USE_DEFAULT,  // quartStart, quartEnd
                    USE_DEFAULT, USE_DEFAULT   // quintStart, quintEnd
                };
                hits = checker->findHits(ZETA5, -0.2636600441662106, 5, getLookupTableFloat(), &loopRanges, doubleHitCount);
                int *result;
                for (int i = 0; i < hits->size(); i++) {
                    result = hits->at(i);
                    checker->printHit(getLookupTableDouble(), result[0], result[1], result[2], result[3], result[4], result[5]);
                    // std::cout << "Hit = " << result[0] << "," << result[1] << "," << result[2] << "," 
                    //           << result[3] << "," << result[4] << "," << result[5] << "," << std::endl;
                }
                delete checker;
                return 0;
            } else if (strcasecmp(argv[1], "megaman2") == 0) {
                std::cout << "Using Positive-OnlyHack" << std::endl;
                checker = new GpuQuinticFirstCheckerPositiveOnly();
                // Create loopRanges: zStart=-5, zEnd=5, all others USE_DEFAULT
                // Format: [zStart, zEnd, yStart, yEnd, xStart, xEnd, cubicStart, cubicEnd, quartStart, quartEnd, quintStart, quintEnd]
                std::vector<int> loopRanges = {
                    -6, 6,  // zStart, zEnd
                    -6, 6,  // yStart, yEnd
                    USE_DEFAULT, USE_DEFAULT,  // xStart, xEnd
                    USE_DEFAULT, USE_DEFAULT,  // cubicStart, cubicEnd
                    USE_DEFAULT, USE_DEFAULT,  // quartStart, quartEnd
                    USE_DEFAULT, USE_DEFAULT   // quintStart, quintEnd
                };
                hits = checker->findHits(ZETA5, -0.2636600441662106, 5, getLookupTableFloat(), &loopRanges, doubleHitCount);
                int *result;
                for (int i = 0; i < hits->size(); i++) {
                    result = hits->at(i);
                    checker->printHit(getLookupTableDouble(), result[0], result[1], result[2], result[3], result[4], result[5]);
                    // std::cout << "Hit = " << result[0] << "," << result[1] << "," << result[2] << "," 
                    //           << result[3] << "," << result[4] << "," << result[5] << "," << std::endl;
                }
                delete checker;
                return 0;
			} else if (strcasecmp(argv[1], "megaman3") == 0) {
                std::cout << "Using Positive-OnlyHackTOPFOUR" << std::endl;
                checker = new GpuQuinticFirstCheckerPositiveOnlyTopFour();
                // Create loopRanges: zStart=-5, zEnd=5, all others USE_DEFAULT
                // Format: [zStart, zEnd, yStart, yEnd, xStart, xEnd, cubicStart, cubicEnd, quartStart, quartEnd, quintStart, quintEnd]
                std::vector<int> loopRanges = {
                    -6, 6,  // zStart, zEnd
                    -6, 6,  // yStart, yEnd
                    USE_DEFAULT, USE_DEFAULT,  // xStart, xEnd
                    USE_DEFAULT, USE_DEFAULT,  // cubicStart, cubicEnd
                    USE_DEFAULT, USE_DEFAULT,  // quartStart, quartEnd
                    USE_DEFAULT, USE_DEFAULT   // quintStart, quintEnd
                };
                hits = checker->findHits(ZETA5, -0.2636600441662106, 5, getLookupTableFloat(), &loopRanges, doubleHitCount);
                int *result;
                for (int i = 0; i < hits->size(); i++) {
                    result = hits->at(i);
                    checker->printHit(getLookupTableDouble(), result[0], result[1], result[2], result[3], result[4], result[5]);
                    // std::cout << "Hit = " << result[0] << "," << result[1] << "," << result[2] << "," 
                    //           << result[3] << "," << result[4] << "," << result[5] << "," << std::endl;
                }
                delete checker;
                return 0;
            } else if (strcasecmp(argv[1], "megaman4") == 0) {
                std::cout << "Using Positive-OnlyHack TOPFIVE (y in launch)" << std::endl;
                checker = new GpuQuinticFirstCheckerPositiveOnlyTopFive();
                std::vector<int> loopRanges = {
                    0, 0,
                    USE_DEFAULT, USE_DEFAULT, // NOT SAME AS OTHERS, MEGAMAN!
                    USE_DEFAULT, USE_DEFAULT,
                    USE_DEFAULT, USE_DEFAULT,
                    USE_DEFAULT, USE_DEFAULT,
                    USE_DEFAULT, USE_DEFAULT
                };
                hits = checker->findHits(ZETA5, -0.2636600441662106, 5, getLookupTableFloat(), &loopRanges, doubleHitCount);
                int *result;
                for (int i = 0; i < hits->size(); i++) {
                    result = hits->at(i);
                    checker->printHit(getLookupTableDouble(), result[0], result[1], result[2], result[3], result[4], result[5]);
                }
                delete checker;
                return 0;
            } else if (strcasecmp(argv[1], "megaman5") == 0) {
                std::cout << "Using Positive-OnlyHack TOPSIX (y+z in launch)" << std::endl;
                checker = new GpuQuinticFirstCheckerPositiveOnlyTopSix();
                std::vector<int> loopRanges = {
                    -6, 6,
                    -6, 6,
                    USE_DEFAULT, USE_DEFAULT,
                    USE_DEFAULT, USE_DEFAULT,
                    USE_DEFAULT, USE_DEFAULT,
                    USE_DEFAULT, USE_DEFAULT
                };
                hits = checker->findHits(ZETA5, -0.2636600441662106, 5, getLookupTableFloat(), &loopRanges, doubleHitCount);
                int *result;
                for (int i = 0; i < hits->size(); i++) {
                    result = hits->at(i);
                    checker->printHit(getLookupTableDouble(), result[0], result[1], result[2], result[3], result[4], result[5]);
                }
                delete checker;
                return 0;
            } else {
                std::cout << "Could not parse checker, using CpuQuinticFirstWithBreakouts" << std::endl;
                checker = new CpuQuinticFirstWithBreakoutsChecker();
            }
            break;
        }
        default:
        {
            std::cout << "No checker specified: using CpuQuinticFirstWithBreakouts" << std::endl;
            checker = new CpuQuinticFirstWithBreakoutsChecker();
            break;
        }
    }

    // Initialize MySQL connection and fetch theConst from database
    MySQLConfig config; // This will read from environment variables and validate them
    MYSQL* mysql = initializeMySQLConnection(config);
    if (!mysql) {
        std::cerr << "Failed to initialize MySQL connection. Exiting." << std::endl;
        return 1;
    }
    
    std::cout << "Connected to MySQL database successfully." << std::endl;

    CubicRootSliceWorker worker(mysql, checker);
    if (!worker.resolveWorkerId()) {
        mysql_close(mysql);
        delete checker;
        return 1;
    }
    if (!worker.runOneCubicRoot()) {
        std::cout << "No cubic-root work to process. Exiting." << std::endl;
        mysql_close(mysql);
        delete checker;
        std::cout << "MySQL connection closed." << std::endl;
        return 0;
    }

    delete checker;
    
    // Clean up MySQL connection
    mysql_close(mysql);
    std::cout << "MySQL connection closed." << std::endl;

    return 0;
}