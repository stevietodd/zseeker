#include <iostream>
#include <string>
#include <strings.h>
#include <mysql/mysql.h>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>
#include <vector>
#include <optional>
#include <unistd.h>
#include "CpuPolynomialChecker.hpp"
#include "GpuPolynomialChecker.hpp"
#include "math.hpp"
#include "lookupTableAccessor.hpp"

#define WORK_ITEMS_PER_QUERY 1

struct SliceWorkItem {
    long long sliceId;
    int cubicRootId;
    int zrootSlot;
    int quintLo;
    int quintHi;
    int quartLo;
    int quartHi;
    std::optional<double> zrootVal;
};

static void freeHitsVector(std::vector<int*>* hits) {
    if (!hits) {
        return;
    }
    for (int* p : *hits) {
        delete[] p;
    }
    delete hits;
}

// GpuQuinticFirstChecker loopRanges order: quint, quart, cubic, x, y, z (12 entries for degree 5).
static std::vector<int> makeQuinticFirstSliceLoopRanges(int quintLo, int quintHi, int quartLo, int quartHi) {
    return {
        quintLo, quintHi,
        quartLo, quartHi,
        USE_DEFAULT, USE_DEFAULT,
        USE_DEFAULT, USE_DEFAULT,
        USE_DEFAULT, USE_DEFAULT,
        USE_DEFAULT, USE_DEFAULT,
        USE_DEFAULT, USE_DEFAULT
    };
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

// Claim pending rows from roots_checked_slice (quint x quart tiles per zroot slot).
std::vector<SliceWorkItem> getSliceWorkToBeDone(MYSQL* mysql) {
    std::vector<SliceWorkItem> items;

    std::string query = "SELECT id FROM roots_checked_slice WHERE is_started = 0 AND is_finished = 0 ORDER BY id ASC LIMIT "
        + std::to_string(WORK_ITEMS_PER_QUERY);

    if (mysql_query(mysql, query.c_str())) {
        std::cerr << "Error: roots_checked_slice query failed: " << mysql_error(mysql) << std::endl;
        return items;
    }

    MYSQL_RES* result = mysql_store_result(mysql);
    if (!result) {
        std::cerr << "Error: mysql_store_result failed: " << mysql_error(mysql) << std::endl;
        return items;
    }

    std::vector<long long> ids;
    MYSQL_ROW row;
    while ((row = mysql_fetch_row(result))) {
        if (row[0]) {
            ids.push_back(std::stoll(row[0]));
        }
    }
    mysql_free_result(result);

    if (ids.empty()) {
        std::cout << "No slice work items with is_started = 0" << std::endl;
        return items;
    }

    char hostnameBuffer[61];
    if (gethostname(hostnameBuffer, sizeof(hostnameBuffer)) != 0) {
        std::cerr << "Error: gethostname() failed" << std::endl;
        return items;
    }
    std::string hostname(hostnameBuffer);

    char escapedHostname[121];
    unsigned long escapedLen = mysql_real_escape_string(mysql, escapedHostname, hostname.c_str(), hostname.length());
    std::string workerQuery = "SELECT id FROM workers WHERE hostname = '" + std::string(escapedHostname, escapedLen) + "' LIMIT 1";
    int workerId = -1;

    if (mysql_query(mysql, workerQuery.c_str())) {
        std::cerr << "Error: workers query failed: " << mysql_error(mysql) << std::endl;
        return items;
    }

    MYSQL_RES* workerResult = mysql_store_result(mysql);
    if (!workerResult) {
        std::cerr << "Error: mysql_store_result failed for workers query: " << mysql_error(mysql) << std::endl;
        return items;
    }

    MYSQL_ROW workerRow = mysql_fetch_row(workerResult);
    if (workerRow && workerRow[0]) {
        workerId = std::stoi(workerRow[0]);
        std::cout << "Found worker_id: " << workerId << std::endl;
    } else {
        std::cerr << "Error: No workers found for hostname = " << hostname << std::endl;
        mysql_free_result(workerResult);
        return items;
    }
    mysql_free_result(workerResult);

    std::string updateQuery = "UPDATE roots_checked_slice SET is_started = 1, worker_id = " + std::to_string(workerId) + " WHERE id IN (";
    for (size_t i = 0; i < ids.size(); ++i) {
        updateQuery += std::to_string(ids[i]);
        if (i < ids.size() - 1) {
            updateQuery += ",";
        }
    }
    updateQuery += ")";

    if (mysql_query(mysql, updateQuery.c_str())) {
        std::cerr << "Error: roots_checked_slice UPDATE failed: " << mysql_error(mysql) << std::endl;
        return items;
    }

    std::cout << "Updated " << ids.size() << " rows in roots_checked_slice to is_started = 1 and worker_id = " << workerId << std::endl;

    std::string selectQuery =
        "SELECT s.id, s.cubic_root_id, s.zroot_slot, s.quint_lo, s.quint_hi, s.quart_lo, s.quart_hi, "
        "z.zroot1, z.zroot2, z.zroot3 FROM roots_checked_slice s "
        "INNER JOIN z3_cubic_roots z ON z.id = s.cubic_root_id WHERE s.id IN (";
    for (size_t i = 0; i < ids.size(); ++i) {
        selectQuery += std::to_string(ids[i]);
        if (i < ids.size() - 1) {
            selectQuery += ",";
        }
    }
    selectQuery += ") ORDER BY s.id";

    if (mysql_query(mysql, selectQuery.c_str())) {
        std::cerr << "Error: slice join query failed: " << mysql_error(mysql) << std::endl;
        return items;
    }

    MYSQL_RES* selectResult = mysql_store_result(mysql);
    if (!selectResult) {
        std::cerr << "Error: mysql_store_result failed: " << mysql_error(mysql) << std::endl;
        return items;
    }

    while ((row = mysql_fetch_row(selectResult))) {
        if (!row[0]) {
            continue;
        }
        SliceWorkItem item;
        item.sliceId = std::stoll(row[0]);
        item.cubicRootId = std::stoi(row[1]);
        item.zrootSlot = std::stoi(row[2]);
        item.quintLo = std::stoi(row[3]);
        item.quintHi = std::stoi(row[4]);
        item.quartLo = std::stoi(row[5]);
        item.quartHi = std::stoi(row[6]);

        std::optional<double> z1 = (row[7] != nullptr) ? std::optional<double>(std::stod(row[7])) : std::nullopt;
        std::optional<double> z2 = (row[8] != nullptr) ? std::optional<double>(std::stod(row[8])) : std::nullopt;
        std::optional<double> z3 = (row[9] != nullptr) ? std::optional<double>(std::stod(row[9])) : std::nullopt;

        if (item.zrootSlot == 1) {
            item.zrootVal = z1;
        } else if (item.zrootSlot == 2) {
            item.zrootVal = z2;
        } else if (item.zrootSlot == 3) {
            item.zrootVal = z3;
        }
        items.push_back(item);
    }
    mysql_free_result(selectResult);

    std::cout << "Fetched " << items.size() << " slice work items" << std::endl;
    return items;
}

static bool updateSliceFinished(MYSQL* mysql, long long sliceId, long doubleHits, long float128Hits) {
    std::string q = "UPDATE roots_checked_slice SET is_finished = 1, double_hit_count = " + std::to_string(doubleHits)
        + ", float128_hit_count = " + std::to_string(float128Hits) + " WHERE id = " + std::to_string(sliceId);
    if (mysql_query(mysql, q.c_str())) {
        std::cerr << "Error: slice finish UPDATE failed: " << mysql_error(mysql) << std::endl;
        return false;
    }
    return true;
}

static bool addSliceHitsToRootsChecked(MYSQL* mysql, int cubicRootId, int zrootSlot, long doubleHits, long float128Hits) {
    if (zrootSlot < 1 || zrootSlot > 3) {
        std::cerr << "Error: invalid zroot_slot " << zrootSlot << std::endl;
        return false;
    }
    std::string dcol = "double_hit_count" + std::to_string(zrootSlot);
    std::string fcol = "float128_hit_count" + std::to_string(zrootSlot);
    std::string q = "UPDATE roots_checked SET " + dcol + " = COALESCE(" + dcol + ", 0) + " + std::to_string(doubleHits)
        + ", " + fcol + " = COALESCE(" + fcol + ", 0) + " + std::to_string(float128Hits) + " WHERE id = " + std::to_string(cubicRootId);
    if (mysql_query(mysql, q.c_str())) {
        std::cerr << "Error: roots_checked incremental UPDATE failed: " << mysql_error(mysql) << std::endl;
        return false;
    }
    return true;
}

static bool finalizeRootsCheckedIfAllSlicesDone(MYSQL* mysql, int cubicRootId) {
    std::string cntQuery = "SELECT COUNT(*) FROM roots_checked_slice WHERE cubic_root_id = " + std::to_string(cubicRootId)
        + " AND is_finished = 0";
    if (mysql_query(mysql, cntQuery.c_str())) {
        std::cerr << "Error: slice count query failed: " << mysql_error(mysql) << std::endl;
        return false;
    }
    MYSQL_RES* res = mysql_store_result(mysql);
    if (!res) {
        std::cerr << "Error: mysql_store_result failed for slice count" << std::endl;
        return false;
    }
    MYSQL_ROW r = mysql_fetch_row(res);
    long long remaining = 0;
    if (r && r[0]) {
        remaining = std::stoll(r[0]);
    }
    mysql_free_result(res);

    if (remaining > 0) {
        return true;
    }

    std::string fin = "UPDATE roots_checked SET is_finished = 1 WHERE id = " + std::to_string(cubicRootId);
    if (mysql_query(mysql, fin.c_str())) {
        std::cerr << "Error: roots_checked finalize UPDATE failed: " << mysql_error(mysql) << std::endl;
        return false;
    }
    std::cout << "All slices done for cubic_root_id " << cubicRootId << "; set roots_checked.is_finished = 1" << std::endl;
    return true;
}

// After CALL populate_slices_for_cubic_root, drain all result sets (required by libmysqlclient).
static bool mysqlDrainProcedureCallResults(MYSQL* mysql) {
    MYSQL_RES* res = mysql_store_result(mysql);
    if (res) {
        mysql_free_result(res);
    }
    int status = 0;
    while ((status = mysql_next_result(mysql)) == 0) {
        res = mysql_store_result(mysql);
        if (res) {
            mysql_free_result(res);
        }
    }
    if (status != -1) {
        std::cerr << "Error: mysql_next_result: " << mysql_error(mysql) << std::endl;
        return false;
    }
    return true;
}

static bool callPopulateSlicesForCubicRoot(MYSQL* mysql, unsigned cubicRootId) {
    std::string q = "CALL populate_slices_for_cubic_root(" + std::to_string(cubicRootId) + ","
        + std::to_string(DEFAULT_SLICE_QUINT_CHUNK) + "," + std::to_string(DEFAULT_SLICE_QUART_CHUNK) + ")";
    if (mysql_query(mysql, q.c_str())) {
        std::cerr << "Error: " << q << " failed: " << mysql_error(mysql) << std::endl;
        return false;
    }
    return mysqlDrainProcedureCallResults(mysql);
}

// For each unfinished roots_checked row with no slice tiles yet, insert 1-quint x 30-quart tiles.
static void ensureSliceRowsForPendingRoots(MYSQL* mysql) {
    std::cout << "Auto-populating roots_checked_slice where missing (quint_chunk="
              << DEFAULT_SLICE_QUINT_CHUNK << ", quart_chunk=" << DEFAULT_SLICE_QUART_CHUNK << ")." << std::endl;

    const char* sel =
        "SELECT id FROM roots_checked WHERE is_finished = 0 AND NOT EXISTS (SELECT 1 FROM roots_checked_slice s "
        "WHERE s.cubic_root_id = roots_checked.id) ORDER BY id";
    if (mysql_query(mysql, sel)) {
        std::cerr << "Error: ensureSliceRows SELECT failed: " << mysql_error(mysql) << std::endl;
        return;
    }
    MYSQL_RES* result = mysql_store_result(mysql);
    if (!result) {
        std::cerr << "Error: mysql_store_result failed for ensureSliceRows" << std::endl;
        return;
    }

    std::vector<unsigned> rootsNeedingSlices;
    MYSQL_ROW row;
    while ((row = mysql_fetch_row(result))) {
        if (row[0]) {
            rootsNeedingSlices.push_back(static_cast<unsigned>(std::stoul(row[0])));
        }
    }
    mysql_free_result(result);

    if (rootsNeedingSlices.empty()) {
        std::cout << "No roots_checked rows need new slice tiles (or table empty)." << std::endl;
        return;
    }

    std::cout << "Populating slices for " << rootsNeedingSlices.size() << " cubic root(s)..." << std::endl;
    for (unsigned rid : rootsNeedingSlices) {
        std::cout << "  populate_slices_for_cubic_root(" << rid << ", " << DEFAULT_SLICE_QUINT_CHUNK
                  << ", " << DEFAULT_SLICE_QUART_CHUNK << ")" << std::endl;
        if (!callPopulateSlicesForCubicRoot(mysql, rid)) {
            std::cerr << "Aborting slice auto-population after failure for cubic_root_id " << rid << std::endl;
            return;
        }
    }
    std::cout << "Slice auto-population finished." << std::endl;
}

static void runOneSlice(
    PolynomialCheckerInterface* checker,
    MYSQL* mysql,
    const SliceWorkItem& s)
{
    std::cout << "\n=== Slice id=" << s.sliceId << " cubic_root_id=" << s.cubicRootId
              << " zroot_slot=" << s.zrootSlot << " quint=[" << s.quintLo << "," << s.quintHi << "] quart=["
              << s.quartLo << "," << s.quartHi << "] ===" << std::endl;

    long sliceDouble = 0;
    long sliceFloat128 = 0;

    if (!s.zrootVal.has_value()) {
        std::cerr << "Warning: zroot is NULL for slice id " << s.sliceId << "; marking finished with 0 hits." << std::endl;
    } else {
        const double theConst = s.zrootVal.value();
        std::cout << "theConst = " << theConst << std::endl;
        std::vector<int> loopRanges = makeQuinticFirstSliceLoopRanges(s.quintLo, s.quintHi, s.quartLo, s.quartHi);
        long doubleHitCount = 0;
        std::vector<int*>* hits = checker->findHits(ZETA5, theConst, 5, getLookupTableFloat(), &loopRanges, doubleHitCount);
        sliceDouble = doubleHitCount;
        sliceFloat128 = static_cast<long>(hits->size());

        for (size_t i = 0; i < hits->size(); i++) {
            int* result = hits->at(i);
            std::cout << "Hit = " << result[0] << "," << result[1] << "," << result[2] << "," << result[3] << "," << result[4] << ","
                      << result[5] << "," << std::endl;
        }
        freeHitsVector(hits);
    }

    if (!updateSliceFinished(mysql, s.sliceId, sliceDouble, sliceFloat128)) {
        return;
    }
    if (!addSliceHitsToRootsChecked(mysql, s.cubicRootId, s.zrootSlot, sliceDouble, sliceFloat128)) {
        return;
    }
    finalizeRootsCheckedIfAllSlicesDone(mysql, s.cubicRootId);
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
    std::cout << "Using 1-quint x 30-quart slices (GpuQuinticFirst loop order)." << std::endl;
    ensureSliceRowsForPendingRoots(mysql);

    std::vector<SliceWorkItem> sliceItems = getSliceWorkToBeDone(mysql);
    if (sliceItems.empty()) {
        std::cout << "No slice work items to process. Exiting." << std::endl;
        mysql_close(mysql);
        delete checker;
        std::cout << "MySQL connection closed." << std::endl;
        return 0;
    }

    for (const SliceWorkItem& s : sliceItems) {
        runOneSlice(checker, mysql, s);
    }

    delete checker;
    
    // Clean up MySQL connection
    mysql_close(mysql);
    std::cout << "MySQL connection closed." << std::endl;

    return 0;
}