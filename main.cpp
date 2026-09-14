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
#include <chrono>
#include <algorithm>
#include <array>
#include <map>
#include <utility>
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

struct LutBounds {
    int quintMin;
    int quintMax;
    int quartMin;
    int quartMax;
};

struct TileRect {
    int qLo;
    int qHi;
    int rLo;
    int rHi;
};

struct SlotPlan {
    int quintChunk;
    int quartChunk;
    LutBounds bounds;
    TileRect probe;
};

struct SliceSearchResult {
    long doubleHits = 0;
    long float128Hits = 0;
};

struct Region {
    int q0;
    int q1;
    int r0;
    int r1;
};

static std::map<std::pair<int, int>, SlotPlan> g_slotPlans;

static LutBounds makeLutBounds(bool positiveOnlyQuint) {
    return {
        positiveOnlyQuint ? 0 : SLICE_QUINT_MIN,
        SLICE_QUINT_MAX,
        SLICE_QUART_MIN,
        SLICE_QUART_MAX
    };
}

static bool checkerUsesPositiveOnlyQuint(PolynomialCheckerInterface* checker) {
    return dynamic_cast<GpuQuinticFirstCheckerPositiveOnly*>(checker) != nullptr
        || dynamic_cast<GpuQuinticFirstCheckerPositiveOnlyTopFour*>(checker) != nullptr
        || dynamic_cast<GpuQuinticFirstCheckerPositiveOnlyTopFive*>(checker) != nullptr
        || dynamic_cast<GpuQuinticFirstCheckerPositiveOnlyTopSix*>(checker) != nullptr;
}

static double targetSliceRuntimeSec() {
    const char* v = getenv("ZSEEKER_SLICE_TARGET_SEC");
    if (v && *v) {
        const double d = std::atof(v);
        if (d > 0) {
            return d;
        }
    }
    return TARGET_SLICE_RUNTIME_SEC;
}

static std::optional<int> lookupWorkerId(MYSQL* mysql) {
    char hostnameBuffer[61];
    if (gethostname(hostnameBuffer, sizeof(hostnameBuffer)) != 0) {
        std::cerr << "Error: gethostname() failed" << std::endl;
        return std::nullopt;
    }
    std::string hostname(hostnameBuffer);

    char escapedHostname[121];
    unsigned long escapedLen = mysql_real_escape_string(mysql, escapedHostname, hostname.c_str(), hostname.length());
    std::string workerQuery = "SELECT id FROM workers WHERE hostname = '" + std::string(escapedHostname, escapedLen) + "' LIMIT 1";

    if (mysql_query(mysql, workerQuery.c_str())) {
        std::cerr << "Error: workers query failed: " << mysql_error(mysql) << std::endl;
        return std::nullopt;
    }

    MYSQL_RES* workerResult = mysql_store_result(mysql);
    if (!workerResult) {
        std::cerr << "Error: mysql_store_result failed for workers query" << std::endl;
        return std::nullopt;
    }

    MYSQL_ROW workerRow = mysql_fetch_row(workerResult);
    std::optional<int> workerId;
    if (workerRow && workerRow[0]) {
        workerId = std::stoi(workerRow[0]);
        std::cout << "Found worker_id: " << *workerId << std::endl;
    } else {
        std::cerr << "Error: No workers found for hostname = " << hostname << std::endl;
    }
    mysql_free_result(workerResult);
    return workerId;
}

static std::array<std::optional<double>, 4> loadZroots(MYSQL* mysql, int cubicRootId) {
    std::array<std::optional<double>, 4> zroots{};
    std::string q = "SELECT zroot1, zroot2, zroot3 FROM z3_cubic_roots WHERE id = " + std::to_string(cubicRootId) + " LIMIT 1";
    if (mysql_query(mysql, q.c_str())) {
        std::cerr << "Error: z3_cubic_roots query failed: " << mysql_error(mysql) << std::endl;
        return zroots;
    }
    MYSQL_RES* result = mysql_store_result(mysql);
    if (!result) {
        std::cerr << "Error: mysql_store_result failed for z3_cubic_roots" << std::endl;
        return zroots;
    }
    MYSQL_ROW row = mysql_fetch_row(result);
    if (row) {
        if (row[0]) {
            zroots[1] = std::stod(row[0]);
        }
        if (row[1]) {
            zroots[2] = std::stod(row[1]);
        }
        if (row[2]) {
            zroots[3] = std::stod(row[2]);
        }
    }
    mysql_free_result(result);
    return zroots;
}

static std::optional<int> fetchSingleInt(MYSQL* mysql, const std::string& q) {
    if (mysql_query(mysql, q.c_str())) {
        std::cerr << "Error: " << q << " failed: " << mysql_error(mysql) << std::endl;
        return std::nullopt;
    }
    MYSQL_RES* res = mysql_store_result(mysql);
    if (!res) {
        std::cerr << "Error: mysql_store_result failed" << std::endl;
        return std::nullopt;
    }
    MYSQL_ROW row = mysql_fetch_row(res);
    std::optional<int> value;
    if (row && row[0]) {
        value = std::stoi(row[0]);
    }
    mysql_free_result(res);
    return value;
}

static std::optional<int> peekPendingSliceRoot(MYSQL* mysql) {
    return fetchSingleInt(mysql,
        "SELECT cubic_root_id FROM roots_checked_slice WHERE is_started = 0 AND is_finished = 0 ORDER BY id ASC LIMIT 1");
}

static std::optional<int> peekStartedRootWithNoSlices(MYSQL* mysql) {
    return fetchSingleInt(mysql,
        "SELECT rc.id FROM roots_checked rc WHERE rc.is_started = 1 AND rc.is_finished = 0 "
        "AND NOT EXISTS (SELECT 1 FROM roots_checked_slice s WHERE s.cubic_root_id = rc.id) "
        "ORDER BY rc.id ASC LIMIT 1");
}

static bool enqueueNextRemainingForRoot(MYSQL* mysql, int cubicRootId, const LutBounds& bounds);

static std::optional<int> enqueueRemainingForAnyUnfinishedRoot(MYSQL* mysql, const LutBounds& bounds) {
    const char* q = "SELECT id FROM roots_checked WHERE is_started = 1 AND is_finished = 0 ORDER BY id ASC";
    if (mysql_query(mysql, q)) {
        std::cerr << "Error: unfinished roots query failed: " << mysql_error(mysql) << std::endl;
        return std::nullopt;
    }
    MYSQL_RES* res = mysql_store_result(mysql);
    if (!res) {
        std::cerr << "Error: mysql_store_result failed for unfinished roots" << std::endl;
        return std::nullopt;
    }
    std::vector<int> ids;
    MYSQL_ROW row;
    while ((row = mysql_fetch_row(res))) {
        if (row[0]) {
            ids.push_back(std::stoi(row[0]));
        }
    }
    mysql_free_result(res);
    for (int id : ids) {
        if (enqueueNextRemainingForRoot(mysql, id, bounds)) {
            return id;
        }
    }
    return std::nullopt;
}

static std::optional<int> claimNewCubicRoot(MYSQL* mysql, int workerId) {
    std::string u = "UPDATE roots_checked SET is_started = 1, worker_id = " + std::to_string(workerId)
        + " WHERE is_started = 0 AND is_finished = 0 AND NOT EXISTS ("
        "SELECT 1 FROM roots_checked_slice s WHERE s.cubic_root_id = roots_checked.id"
        ") ORDER BY id ASC LIMIT 1";
    if (mysql_query(mysql, u.c_str())) {
        std::cerr << "Error: claim cubic root UPDATE failed: " << mysql_error(mysql) << std::endl;
        return std::nullopt;
    }
    if (mysql_affected_rows(mysql) == 0) {
        return std::nullopt;
    }
    return fetchSingleInt(mysql,
        "SELECT id FROM roots_checked WHERE worker_id = " + std::to_string(workerId)
        + " AND is_started = 1 AND is_finished = 0 AND NOT EXISTS ("
        "SELECT 1 FROM roots_checked_slice s WHERE s.cubic_root_id = roots_checked.id"
        ") ORDER BY id ASC LIMIT 1");
}

static bool loadLastTile(MYSQL* mysql, int cubicRootId, int slot, TileRect& out) {
    std::string q = "SELECT quint_lo, quint_hi, quart_lo, quart_hi FROM roots_checked_slice WHERE cubic_root_id = "
        + std::to_string(cubicRootId) + " AND zroot_slot = " + std::to_string(slot)
        + " ORDER BY quint_lo DESC, quart_lo DESC LIMIT 1";
    if (mysql_query(mysql, q.c_str())) {
        std::cerr << "Error: last-tile query failed: " << mysql_error(mysql) << std::endl;
        return false;
    }
    MYSQL_RES* res = mysql_store_result(mysql);
    if (!res) {
        std::cerr << "Error: mysql_store_result failed for last-tile query" << std::endl;
        return false;
    }
    MYSQL_ROW row = mysql_fetch_row(res);
    const bool ok = row && row[0] && row[1] && row[2] && row[3];
    if (ok) {
        out.qLo = std::stoi(row[0]);
        out.qHi = std::stoi(row[1]);
        out.rLo = std::stoi(row[2]);
        out.rHi = std::stoi(row[3]);
    }
    mysql_free_result(res);
    return ok;
}

static bool loadFirstTile(MYSQL* mysql, int cubicRootId, int slot, TileRect& out, long long& sliceId) {
    std::string q = "SELECT id, quint_lo, quint_hi, quart_lo, quart_hi FROM roots_checked_slice WHERE cubic_root_id = "
        + std::to_string(cubicRootId) + " AND zroot_slot = " + std::to_string(slot)
        + " ORDER BY id ASC LIMIT 1";
    if (mysql_query(mysql, q.c_str())) {
        std::cerr << "Error: first-tile query failed: " << mysql_error(mysql) << std::endl;
        return false;
    }
    MYSQL_RES* res = mysql_store_result(mysql);
    if (!res) {
        std::cerr << "Error: mysql_store_result failed for first-tile query" << std::endl;
        return false;
    }
    MYSQL_ROW row = mysql_fetch_row(res);
    const bool ok = row && row[0] && row[1] && row[2] && row[3] && row[4];
    if (ok) {
        sliceId = std::stoll(row[0]);
        out.qLo = std::stoi(row[1]);
        out.qHi = std::stoi(row[2]);
        out.rLo = std::stoi(row[3]);
        out.rHi = std::stoi(row[4]);
    }
    mysql_free_result(res);
    return ok;
}

static bool loadFirstNonProbeTile(MYSQL* mysql, int cubicRootId, int slot, long long probeId, TileRect& out) {
    std::string q = "SELECT quint_lo, quint_hi, quart_lo, quart_hi FROM roots_checked_slice WHERE cubic_root_id = "
        + std::to_string(cubicRootId) + " AND zroot_slot = " + std::to_string(slot)
        + " AND id <> " + std::to_string(probeId) + " ORDER BY id ASC LIMIT 1";
    if (mysql_query(mysql, q.c_str())) {
        std::cerr << "Error: non-probe tile query failed: " << mysql_error(mysql) << std::endl;
        return false;
    }
    MYSQL_RES* res = mysql_store_result(mysql);
    if (!res) {
        std::cerr << "Error: mysql_store_result failed for non-probe tile query" << std::endl;
        return false;
    }
    MYSQL_ROW row = mysql_fetch_row(res);
    const bool ok = row && row[0] && row[1] && row[2] && row[3];
    if (ok) {
        out.qLo = std::stoi(row[0]);
        out.qHi = std::stoi(row[1]);
        out.rLo = std::stoi(row[2]);
        out.rHi = std::stoi(row[3]);
    }
    mysql_free_result(res);
    return ok;
}

static bool sameTile(const TileRect& a, const TileRect& b) {
    return a.qLo == b.qLo && a.qHi == b.qHi && a.rLo == b.rLo && a.rHi == b.rHi;
}

static bool inferPlanFromDb(MYSQL* mysql, int cubicRootId, int slot, const LutBounds& bounds, SlotPlan& plan) {
    TileRect probe{};
    long long probeId = 0;
    if (!loadFirstTile(mysql, cubicRootId, slot, probe, probeId)) {
        return false;
    }
    plan.bounds = bounds;
    plan.probe = probe;
    plan.quintChunk = DEFAULT_SLICE_QUINT_CHUNK;
    plan.quartChunk = DEFAULT_SLICE_QUART_CHUNK;
    TileRect later{};
    if (loadFirstNonProbeTile(mysql, cubicRootId, slot, probeId, later)) {
        plan.quintChunk = std::max(1, later.qHi - later.qLo + 1);
        plan.quartChunk = std::max(DEFAULT_SLICE_QUART_CHUNK, later.rHi - later.rLo + 1);
    }
    return true;
}

static bool planForSlot(MYSQL* mysql, int cubicRootId, int slot, const LutBounds& bounds, SlotPlan& plan) {
    const auto key = std::make_pair(cubicRootId, slot);
    auto it = g_slotPlans.find(key);
    if (it != g_slotPlans.end()) {
        plan = it->second;
        return true;
    }
    if (!inferPlanFromDb(mysql, cubicRootId, slot, bounds, plan)) {
        return false;
    }
    g_slotPlans[key] = plan;
    return true;
}

static std::array<Region, 2> remainingRegions(const SlotPlan& p) {
    return {{
        {p.probe.qLo, p.probe.qHi, p.probe.rHi + 1, p.bounds.quartMax},
        {p.probe.qHi + 1, p.bounds.quintMax, p.bounds.quartMin, p.bounds.quartMax}
    }};
}

static bool regionNonEmpty(const Region& reg) {
    return reg.q0 <= reg.q1 && reg.r0 <= reg.r1;
}

static bool firstTileInRegion(const SlotPlan& p, const Region& reg, TileRect& out) {
    if (!regionNonEmpty(reg)) {
        return false;
    }
    out.qLo = reg.q0;
    out.qHi = std::min(reg.q0 + p.quintChunk - 1, reg.q1);
    out.rLo = reg.r0;
    out.rHi = std::min(reg.r0 + p.quartChunk - 1, reg.r1);
    return true;
}

static bool nextTileAfterInRegion(const SlotPlan& p, const Region& reg, const TileRect& last, TileRect& out) {
    const int nextR = last.rHi + 1;
    if (nextR <= reg.r1) {
        out.qLo = last.qLo;
        out.qHi = std::min(last.qLo + p.quintChunk - 1, reg.q1);
        out.rLo = nextR;
        out.rHi = std::min(nextR + p.quartChunk - 1, reg.r1);
        return true;
    }
    const int nextQ = last.qHi + 1;
    if (nextQ <= reg.q1) {
        out.qLo = nextQ;
        out.qHi = std::min(nextQ + p.quintChunk - 1, reg.q1);
        out.rLo = reg.r0;
        out.rHi = std::min(reg.r0 + p.quartChunk - 1, reg.r1);
        return true;
    }
    return false;
}

static int regionIndexOfRemaining(const SlotPlan& p, const TileRect& t) {
    if (t.qLo >= p.probe.qLo && t.qLo <= p.probe.qHi && t.rLo > p.probe.rHi) {
        return 0;
    }
    if (t.qLo > p.probe.qHi) {
        return 1;
    }
    return -1;
}

static bool nextRemainingTiles(const SlotPlan& p, const TileRect& last, bool lastIsProbe, int maxN, std::vector<TileRect>& out) {
    out.clear();
    const auto regs = remainingRegions(p);
    int ri = 0;
    TileRect cursor{};

    auto seekFirstFrom = [&](int startRi) {
        ri = startRi;
        while (ri < 2 && !firstTileInRegion(p, regs[ri], cursor)) {
            ++ri;
        }
        return ri < 2;
    };

    if (lastIsProbe) {
        if (!seekFirstFrom(0)) {
            return false;
        }
        out.push_back(cursor);
    } else {
        ri = regionIndexOfRemaining(p, last);
        if (ri < 0) {
            if (!seekFirstFrom(0)) {
                return false;
            }
            out.push_back(cursor);
        } else if (nextTileAfterInRegion(p, regs[ri], last, cursor)) {
            out.push_back(cursor);
        } else if (!seekFirstFrom(ri + 1)) {
            return false;
        } else {
            out.push_back(cursor);
        }
    }

    while (static_cast<int>(out.size()) < maxN) {
        TileRect n{};
        if (nextTileAfterInRegion(p, regs[ri], out.back(), n)) {
            out.push_back(n);
            continue;
        }
        ++ri;
        if (ri >= 2 || !firstTileInRegion(p, regs[ri], n)) {
            break;
        }
        out.push_back(n);
    }
    return !out.empty();
}

static std::pair<int, int> chunksFromProbe(double probeSec, const LutBounds& bounds) {
    const long long probeCells = static_cast<long long>(DEFAULT_SLICE_QUINT_CHUNK) * DEFAULT_SLICE_QUART_CHUNK;
    if (probeSec < 1e-9) {
        probeSec = 1e-9;
    }
    const double target = targetSliceRuntimeSec();
    long long cells = static_cast<long long>(std::llround(target / (probeSec / static_cast<double>(probeCells))));
    if (cells < probeCells) {
        cells = probeCells;
    }

    const int quartSpan = bounds.quartMax - bounds.quartMin + 1;
    const int quintSpan = bounds.quintMax - bounds.quintMin + 1;
    const int maxQuart = std::min(quartSpan, SLICE_MAX_QUART_RANGE);
    const int maxQuint = std::min(quintSpan, SLICE_MAX_QUINT_RANGE);

    int quartChunk = DEFAULT_SLICE_QUART_CHUNK;
    int quintChunk = DEFAULT_SLICE_QUINT_CHUNK;
    if (cells <= maxQuart) {
        quartChunk = static_cast<int>(cells);
        quintChunk = 1;
    } else {
        quartChunk = maxQuart;
        long long q = (cells + maxQuart - 1) / maxQuart;
        if (q < 1) {
            q = 1;
        }
        if (q > maxQuint) {
            q = maxQuint;
        }
        quintChunk = static_cast<int>(q);
    }
    return {quintChunk, quartChunk};
}

static std::optional<long long> insertSliceRow(
    MYSQL* mysql,
    int cubicRootId,
    int slot,
    const TileRect& t,
    bool started,
    std::optional<int> workerId)
{
    std::string q = "INSERT IGNORE INTO roots_checked_slice (cubic_root_id, zroot_slot, quint_lo, quint_hi, quart_lo, quart_hi, worker_id, is_started) VALUES ("
        + std::to_string(cubicRootId) + "," + std::to_string(slot) + ","
        + std::to_string(t.qLo) + "," + std::to_string(t.qHi) + ","
        + std::to_string(t.rLo) + "," + std::to_string(t.rHi) + ",";
    if (workerId.has_value()) {
        q += std::to_string(*workerId);
    } else {
        q += "NULL";
    }
    q += "," + std::to_string(started ? 1 : 0) + ")";
    if (mysql_query(mysql, q.c_str())) {
        std::cerr << "Error: slice INSERT failed: " << mysql_error(mysql) << std::endl;
        return std::nullopt;
    }
    if (mysql_affected_rows(mysql) == 0) {
        return 0;
    }
    return static_cast<long long>(mysql_insert_id(mysql));
}

static bool insertSliceBatchUnstarted(MYSQL* mysql, int cubicRootId, int slot, const std::vector<TileRect>& tiles) {
    if (tiles.empty()) {
        return true;
    }
    std::string q = "INSERT IGNORE INTO roots_checked_slice (cubic_root_id, zroot_slot, quint_lo, quint_hi, quart_lo, quart_hi) VALUES ";
    for (size_t i = 0; i < tiles.size(); ++i) {
        const TileRect& t = tiles[i];
        if (i > 0) {
            q += ",";
        }
        q += "(" + std::to_string(cubicRootId) + "," + std::to_string(slot) + ","
            + std::to_string(t.qLo) + "," + std::to_string(t.qHi) + ","
            + std::to_string(t.rLo) + "," + std::to_string(t.rHi) + ")";
    }
    if (mysql_query(mysql, q.c_str())) {
        std::cerr << "Error: slice batch INSERT failed: " << mysql_error(mysql) << std::endl;
        return false;
    }
    return true;
}

static bool enqueueNextRemainingForRoot(MYSQL* mysql, int cubicRootId, const LutBounds& bounds) {
    bool any = false;
    for (int slot = 1; slot <= 3; ++slot) {
        SlotPlan plan{};
        if (!planForSlot(mysql, cubicRootId, slot, bounds, plan)) {
            continue;
        }
        TileRect last{};
        if (!loadLastTile(mysql, cubicRootId, slot, last)) {
            continue;
        }
        const bool lastIsProbe = sameTile(last, plan.probe);
        std::vector<TileRect> tiles;
        if (!nextRemainingTiles(plan, last, lastIsProbe, SLICE_ENQUEUE_BATCH, tiles)) {
            continue;
        }
        if (!insertSliceBatchUnstarted(mysql, cubicRootId, slot, tiles)) {
            return false;
        }
        std::cout << "Enqueued " << tiles.size() << " remaining slice(s) for cubic_root_id " << cubicRootId
                  << " zroot_slot=" << slot << " at " << plan.quintChunk << "x" << plan.quartChunk << std::endl;
        any = true;
    }
    return any;
}

// Claim pending rows from roots_checked_slice (quint x quart tiles per zroot slot).
std::vector<SliceWorkItem> getSliceWorkToBeDone(MYSQL* mysql, int workerId, std::optional<int> cubicRootId = std::nullopt) {
    std::vector<SliceWorkItem> items;

    std::string query = "SELECT id FROM roots_checked_slice WHERE is_started = 0 AND is_finished = 0";
    if (cubicRootId.has_value()) {
        query += " AND cubic_root_id = " + std::to_string(*cubicRootId);
    }
    query += " ORDER BY id ASC LIMIT " + std::to_string(WORK_ITEMS_PER_QUERY);

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
        return items;
    }

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
    std::cout << "All slices currently in the table for cubic_root_id " << cubicRootId
              << " are finished; set roots_checked.is_finished = 1" << std::endl;
    return true;
}

static SliceSearchResult executeSliceSearch(PolynomialCheckerInterface* checker, const SliceWorkItem& s) {
    SliceSearchResult result;
    std::cout << "\n=== Slice id=" << s.sliceId << " cubic_root_id=" << s.cubicRootId
              << " zroot_slot=" << s.zrootSlot << " quint=[" << s.quintLo << "," << s.quintHi << "] quart=["
              << s.quartLo << "," << s.quartHi << "] ===" << std::endl;

    if (!s.zrootVal.has_value()) {
        std::cerr << "Warning: zroot is NULL for slice id " << s.sliceId << "; marking finished with 0 hits." << std::endl;
        return result;
    }

    const double theConst = s.zrootVal.value();
    std::cout << "theConst = " << theConst << std::endl;
    std::vector<int> loopRanges = makeQuinticFirstSliceLoopRanges(s.quintLo, s.quintHi, s.quartLo, s.quartHi);
    long doubleHitCount = 0;
    std::vector<int*>* hits = checker->findHits(ZETA5, theConst, 5, getLookupTableFloat(), &loopRanges, doubleHitCount);
    result.doubleHits = doubleHitCount;
    result.float128Hits = static_cast<long>(hits->size());

    for (size_t i = 0; i < hits->size(); i++) {
        int* hit = hits->at(i);
        std::cout << "Hit = " << hit[0] << "," << hit[1] << "," << hit[2] << "," << hit[3] << "," << hit[4] << ","
                  << hit[5] << "," << std::endl;
    }
    freeHitsVector(hits);
    return result;
}

static bool persistSliceResult(MYSQL* mysql, const SliceWorkItem& s, const SliceSearchResult& result) {
    if (!updateSliceFinished(mysql, s.sliceId, result.doubleHits, result.float128Hits)) {
        return false;
    }
    return addSliceHitsToRootsChecked(mysql, s.cubicRootId, s.zrootSlot, result.doubleHits, result.float128Hits);
}

static bool slotHasSlices(MYSQL* mysql, int cubicRootId, int slot) {
    return fetchSingleInt(mysql,
        "SELECT 1 FROM roots_checked_slice WHERE cubic_root_id = " + std::to_string(cubicRootId)
        + " AND zroot_slot = " + std::to_string(slot) + " LIMIT 1").has_value();
}

static bool probeZrootSlot(
    PolynomialCheckerInterface* checker,
    MYSQL* mysql,
    int cubicRootId,
    int slot,
    double zrootVal,
    int workerId,
    const LutBounds& bounds)
{
    const TileRect probe{
        bounds.quintMin,
        bounds.quintMin + DEFAULT_SLICE_QUINT_CHUNK - 1,
        bounds.quartMin,
        bounds.quartMin + DEFAULT_SLICE_QUART_CHUNK - 1
    };
    const std::optional<long long> sliceId = insertSliceRow(mysql, cubicRootId, slot, probe, true, workerId);
    if (!sliceId.has_value()) {
        return false;
    }
    if (*sliceId == 0) {
        std::cout << "Probe tile already exists for cubic_root_id " << cubicRootId << " zroot_slot=" << slot << std::endl;
        return true;
    }

    SliceWorkItem item;
    item.sliceId = *sliceId;
    item.cubicRootId = cubicRootId;
    item.zrootSlot = slot;
    item.quintLo = probe.qLo;
    item.quintHi = probe.qHi;
    item.quartLo = probe.rLo;
    item.quartHi = probe.rHi;
    item.zrootVal = zrootVal;

    const auto t0 = std::chrono::steady_clock::now();
    const SliceSearchResult result = executeSliceSearch(checker, item);
    const double probeSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!persistSliceResult(mysql, item, result)) {
        return false;
    }

    const auto chunks = chunksFromProbe(probeSec, bounds);
    SlotPlan plan;
    plan.quintChunk = chunks.first;
    plan.quartChunk = chunks.second;
    plan.bounds = bounds;
    plan.probe = probe;
    g_slotPlans[std::make_pair(cubicRootId, slot)] = plan;

    std::cout << "Probe for cubic_root_id " << cubicRootId << " zroot_slot=" << slot << " took " << probeSec
              << "s; remaining tiles " << plan.quintChunk << "x" << plan.quartChunk << " (target "
              << targetSliceRuntimeSec() << "s per tile)" << std::endl;
    return true;
}

static void runOneSlice(PolynomialCheckerInterface* checker, MYSQL* mysql, const SliceWorkItem& s);

static void processCubicRoot(
    PolynomialCheckerInterface* checker,
    MYSQL* mysql,
    int cubicRootId,
    int workerId,
    const LutBounds& bounds)
{
    std::cout << "Processing cubic_root_id " << cubicRootId
              << "; this worker will keep claiming its slices until the root is done." << std::endl;

    const auto zroots = loadZroots(mysql, cubicRootId);
    bool anySlot = false;
    for (int slot = 1; slot <= 3; ++slot) {
        if (!zroots[slot].has_value()) {
            continue;
        }
        anySlot = true;
        if (!slotHasSlices(mysql, cubicRootId, slot)) {
            if (!probeZrootSlot(checker, mysql, cubicRootId, slot, zroots[slot].value(), workerId, bounds)) {
                return;
            }
        }
    }

    if (!anySlot) {
        std::cout << "No zroot values for cubic_root_id " << cubicRootId << "; marking finished." << std::endl;
        finalizeRootsCheckedIfAllSlicesDone(mysql, cubicRootId);
        std::string fin = "UPDATE roots_checked SET is_finished = 1 WHERE id = " + std::to_string(cubicRootId);
        if (mysql_query(mysql, fin.c_str())) {
            std::cerr << "Error: empty-root finalize UPDATE failed: " << mysql_error(mysql) << std::endl;
        }
        return;
    }

    for (;;) {
        const std::vector<SliceWorkItem> items = getSliceWorkToBeDone(mysql, workerId, cubicRootId);
        if (!items.empty()) {
            for (const SliceWorkItem& s : items) {
                runOneSlice(checker, mysql, s);
            }
            continue;
        }
        if (enqueueNextRemainingForRoot(mysql, cubicRootId, bounds)) {
            continue;
        }
        break;
    }
    finalizeRootsCheckedIfAllSlicesDone(mysql, cubicRootId);
}

static void runOneSlice(
    PolynomialCheckerInterface* checker,
    MYSQL* mysql,
    const SliceWorkItem& s)
{
    persistSliceResult(mysql, s, executeSliceSearch(checker, s));
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

    const std::optional<int> workerIdOpt = lookupWorkerId(mysql);
    if (!workerIdOpt.has_value()) {
        mysql_close(mysql);
        delete checker;
        return 1;
    }
    const int workerId = *workerIdOpt;
    const LutBounds bounds = makeLutBounds(checkerUsesPositiveOnlyQuint(checker));
    std::cout << "Cubic-root work: " << DEFAULT_SLICE_QUINT_CHUNK << "x" << DEFAULT_SLICE_QUART_CHUNK
              << " probe, then remaining tiles sized toward " << targetSliceRuntimeSec()
              << "s (quint LUT [" << bounds.quintMin << "," << bounds.quintMax << "])." << std::endl;

    std::optional<int> rootId = peekPendingSliceRoot(mysql);
    if (!rootId.has_value()) {
        rootId = peekStartedRootWithNoSlices(mysql);
    }
    if (!rootId.has_value()) {
        rootId = claimNewCubicRoot(mysql, workerId);
    }
    if (!rootId.has_value()) {
        rootId = enqueueRemainingForAnyUnfinishedRoot(mysql, bounds);
    }
    if (!rootId.has_value()) {
        std::cout << "No cubic-root work to process. Exiting." << std::endl;
        mysql_close(mysql);
        delete checker;
        std::cout << "MySQL connection closed." << std::endl;
        return 0;
    }

    processCubicRoot(checker, mysql, *rootId, workerId, bounds);

    delete checker;
    
    // Clean up MySQL connection
    mysql_close(mysql);
    std::cout << "MySQL connection closed." << std::endl;

    return 0;
}