#include "CubicRootSliceWorker.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#include "GpuPolynomialChecker.hpp"
#include "lookupTableAccessor.hpp"
#include "math.hpp"
#include "sliceTiling.hpp"

namespace {
constexpr int kWorkItemsPerQuery = 1;
}

struct CubicRootSliceWorker::Impl {
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

    struct SliceSearchResult {
        long doubleHits = 0;
        long float128Hits = 0;
    };

    MYSQL* mysql_;
    PolynomialCheckerInterface* checker_;
    int workerId_ = -1;
    LutBounds bounds_{};
    std::map<std::pair<int, int>, SlotPlan> slotPlans_;

    Impl(MYSQL* mysql, PolynomialCheckerInterface* checker);
    bool resolveWorkerId();
    bool runOneCubicRoot();

    static bool checkerUsesPositiveOnlyQuint(PolynomialCheckerInterface* checker);
    static double targetSliceRuntimeSec();
    static void freeHitsVector(std::vector<int*>* hits);

    std::optional<int> lookupWorkerId();
    std::array<std::optional<double>, 4> loadZroots(int cubicRootId);
    std::optional<int> fetchSingleInt(const std::string& q);
    std::optional<int> peekPendingSliceRoot();
    std::optional<int> peekStartedRootWithNoSlices();
    std::optional<int> enqueueRemainingForAnyUnfinishedRoot();
    std::optional<int> claimNewCubicRoot();
    bool loadLastTile(int cubicRootId, int slot, TileRect& out);
    bool loadFirstTile(int cubicRootId, int slot, TileRect& out, long long& sliceId);
    bool loadFirstNonProbeTile(int cubicRootId, int slot, long long probeId, TileRect& out);
    bool inferPlanFromDb(int cubicRootId, int slot, SlotPlan& plan);
    bool planForSlot(int cubicRootId, int slot, SlotPlan& plan);
    std::optional<long long> insertSliceRow(int cubicRootId, int slot, const TileRect& t, bool started, std::optional<int> workerId);
    bool insertSliceBatchUnstarted(int cubicRootId, int slot, const std::vector<TileRect>& tiles);
    bool enqueueNextRemainingForRoot(int cubicRootId);
    std::vector<SliceWorkItem> getSliceWorkToBeDone(std::optional<int> cubicRootId);
    bool updateSliceFinished(long long sliceId, long doubleHits, long float128Hits);
    bool addSliceHitsToRootsChecked(int cubicRootId, int zrootSlot, long doubleHits, long float128Hits);
    bool finalizeRootsCheckedIfAllSlicesDone(int cubicRootId);
    SliceSearchResult executeSliceSearch(const SliceWorkItem& s);
    bool persistSliceResult(const SliceWorkItem& s, const SliceSearchResult& result);
    bool slotHasSlices(int cubicRootId, int slot);
    bool probeZrootSlot(int cubicRootId, int slot, double zrootVal);
    void processCubicRoot(int cubicRootId);
    void runOneSlice(const SliceWorkItem& s);
};


CubicRootSliceWorker::Impl::Impl(MYSQL* mysql, PolynomialCheckerInterface* checker)
    : mysql_(mysql)
    , checker_(checker)
    , bounds_(lutBoundsForChecker(checkerUsesPositiveOnlyQuint(checker)))
{
}

bool CubicRootSliceWorker::Impl::resolveWorkerId() {
    const std::optional<int> workerId = lookupWorkerId();
    if (!workerId.has_value()) {
        return false;
    }
    workerId_ = *workerId;
    std::cout << "Cubic-root work: " << DEFAULT_SLICE_QUINT_CHUNK << "x" << DEFAULT_SLICE_QUART_CHUNK
              << " probe, then remaining tiles sized toward " << targetSliceRuntimeSec()
              << "s (quint LUT [" << bounds_.quintMin << "," << bounds_.quintMax << "])." << std::endl;
    return true;
}

bool CubicRootSliceWorker::Impl::runOneCubicRoot() {
    std::optional<int> rootId = peekPendingSliceRoot();
    if (!rootId.has_value()) {
        rootId = peekStartedRootWithNoSlices();
    }
    if (!rootId.has_value()) {
        rootId = claimNewCubicRoot();
    }
    if (!rootId.has_value()) {
        rootId = enqueueRemainingForAnyUnfinishedRoot();
    }
    if (!rootId.has_value()) {
        return false;
    }
    processCubicRoot(*rootId);
    return true;
}

bool CubicRootSliceWorker::Impl::checkerUsesPositiveOnlyQuint(PolynomialCheckerInterface* checker) {
    return dynamic_cast<GpuQuinticFirstCheckerPositiveOnly*>(checker) != nullptr
        || dynamic_cast<GpuQuinticFirstCheckerPositiveOnlyTopFour*>(checker) != nullptr
        || dynamic_cast<GpuQuinticFirstCheckerPositiveOnlyTopFive*>(checker) != nullptr
        || dynamic_cast<GpuQuinticFirstCheckerPositiveOnlyTopSix*>(checker) != nullptr;
}

double CubicRootSliceWorker::Impl::targetSliceRuntimeSec() {
    const char* v = getenv("ZSEEKER_SLICE_TARGET_SEC");
    if (v && *v) {
        const double d = std::atof(v);
        if (d > 0) {
            return d;
        }
    }
    return TARGET_SLICE_RUNTIME_SEC;
}

std::optional<int> CubicRootSliceWorker::Impl::lookupWorkerId() {
    char hostnameBuffer[61];
    if (gethostname(hostnameBuffer, sizeof(hostnameBuffer)) != 0) {
        std::cerr << "Error: gethostname() failed" << std::endl;
        return std::nullopt;
    }
    std::string hostname(hostnameBuffer);

    char escapedHostname[121];
    unsigned long escapedLen = mysql_real_escape_string(mysql_, escapedHostname, hostname.c_str(), hostname.length());
    std::string workerQuery = "SELECT id FROM workers WHERE hostname = '" + std::string(escapedHostname, escapedLen) + "' LIMIT 1";

    if (mysql_query(mysql_, workerQuery.c_str())) {
        std::cerr << "Error: workers query failed: " << mysql_error(mysql_) << std::endl;
        return std::nullopt;
    }

    MYSQL_RES* workerResult = mysql_store_result(mysql_);
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

std::array<std::optional<double>, 4> CubicRootSliceWorker::Impl::loadZroots(int cubicRootId) {
    std::array<std::optional<double>, 4> zroots{};
    std::string q = "SELECT zroot1, zroot2, zroot3 FROM z3_cubic_roots WHERE id = " + std::to_string(cubicRootId) + " LIMIT 1";
    if (mysql_query(mysql_, q.c_str())) {
        std::cerr << "Error: z3_cubic_roots query failed: " << mysql_error(mysql_) << std::endl;
        return zroots;
    }
    MYSQL_RES* result = mysql_store_result(mysql_);
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

std::optional<int> CubicRootSliceWorker::Impl::fetchSingleInt(const std::string& q) {
    if (mysql_query(mysql_, q.c_str())) {
        std::cerr << "Error: " << q << " failed: " << mysql_error(mysql_) << std::endl;
        return std::nullopt;
    }
    MYSQL_RES* res = mysql_store_result(mysql_);
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

std::optional<int> CubicRootSliceWorker::Impl::peekPendingSliceRoot() {
    return fetchSingleInt(
        "SELECT cubic_root_id FROM roots_checked_slice WHERE is_started = 0 AND is_finished = 0 ORDER BY id ASC LIMIT 1");
}

std::optional<int> CubicRootSliceWorker::Impl::peekStartedRootWithNoSlices() {
    return fetchSingleInt(
        "SELECT rc.id FROM roots_checked rc WHERE rc.is_started = 1 AND rc.is_finished = 0 "
        "AND NOT EXISTS (SELECT 1 FROM roots_checked_slice s WHERE s.cubic_root_id = rc.id) "
        "ORDER BY rc.id ASC LIMIT 1");
}

std::optional<int> CubicRootSliceWorker::Impl::enqueueRemainingForAnyUnfinishedRoot() {
    const char* q = "SELECT id FROM roots_checked WHERE is_started = 1 AND is_finished = 0 ORDER BY id ASC";
    if (mysql_query(mysql_, q)) {
        std::cerr << "Error: unfinished roots query failed: " << mysql_error(mysql_) << std::endl;
        return std::nullopt;
    }
    MYSQL_RES* res = mysql_store_result(mysql_);
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
        if (enqueueNextRemainingForRoot(id)) {
            return id;
        }
    }
    return std::nullopt;
}

std::optional<int> CubicRootSliceWorker::Impl::claimNewCubicRoot() {
    std::string u = "UPDATE roots_checked SET is_started = 1, worker_id = " + std::to_string(workerId_)
        + " WHERE is_started = 0 AND is_finished = 0 AND NOT EXISTS ("
        "SELECT 1 FROM roots_checked_slice s WHERE s.cubic_root_id = roots_checked.id"
        ") ORDER BY id ASC LIMIT 1";
    if (mysql_query(mysql_, u.c_str())) {
        std::cerr << "Error: claim cubic root UPDATE failed: " << mysql_error(mysql_) << std::endl;
        return std::nullopt;
    }
    if (mysql_affected_rows(mysql_) == 0) {
        return std::nullopt;
    }
    return fetchSingleInt(
        "SELECT id FROM roots_checked WHERE worker_id = " + std::to_string(workerId_)
        + " AND is_started = 1 AND is_finished = 0 AND NOT EXISTS ("
        "SELECT 1 FROM roots_checked_slice s WHERE s.cubic_root_id = roots_checked.id"
        ") ORDER BY id ASC LIMIT 1");
}

bool CubicRootSliceWorker::Impl::loadLastTile(int cubicRootId, int slot, TileRect& out) {
    std::string q = "SELECT quint_lo, quint_hi, quart_lo, quart_hi FROM roots_checked_slice WHERE cubic_root_id = "
        + std::to_string(cubicRootId) + " AND zroot_slot = " + std::to_string(slot)
        + " ORDER BY quint_lo DESC, quart_lo DESC LIMIT 1";
    if (mysql_query(mysql_, q.c_str())) {
        std::cerr << "Error: last-tile query failed: " << mysql_error(mysql_) << std::endl;
        return false;
    }
    MYSQL_RES* res = mysql_store_result(mysql_);
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

bool CubicRootSliceWorker::Impl::loadFirstTile(int cubicRootId, int slot, TileRect& out, long long& sliceId) {
    std::string q = "SELECT id, quint_lo, quint_hi, quart_lo, quart_hi FROM roots_checked_slice WHERE cubic_root_id = "
        + std::to_string(cubicRootId) + " AND zroot_slot = " + std::to_string(slot)
        + " ORDER BY id ASC LIMIT 1";
    if (mysql_query(mysql_, q.c_str())) {
        std::cerr << "Error: first-tile query failed: " << mysql_error(mysql_) << std::endl;
        return false;
    }
    MYSQL_RES* res = mysql_store_result(mysql_);
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

bool CubicRootSliceWorker::Impl::loadFirstNonProbeTile(int cubicRootId, int slot, long long probeId, TileRect& out) {
    std::string q = "SELECT quint_lo, quint_hi, quart_lo, quart_hi FROM roots_checked_slice WHERE cubic_root_id = "
        + std::to_string(cubicRootId) + " AND zroot_slot = " + std::to_string(slot)
        + " AND id <> " + std::to_string(probeId) + " ORDER BY id ASC LIMIT 1";
    if (mysql_query(mysql_, q.c_str())) {
        std::cerr << "Error: non-probe tile query failed: " << mysql_error(mysql_) << std::endl;
        return false;
    }
    MYSQL_RES* res = mysql_store_result(mysql_);
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

bool CubicRootSliceWorker::Impl::inferPlanFromDb(int cubicRootId, int slot, SlotPlan& plan) {
    TileRect probe{};
    long long probeId = 0;
    if (!loadFirstTile(cubicRootId, slot, probe, probeId)) {
        return false;
    }
    plan.bounds = bounds_;
    plan.probe = probe;
    plan.quintChunk = DEFAULT_SLICE_QUINT_CHUNK;
    plan.quartChunk = DEFAULT_SLICE_QUART_CHUNK;
    TileRect later{};
    if (loadFirstNonProbeTile(cubicRootId, slot, probeId, later)) {
        plan.quintChunk = std::max(1, later.qHi - later.qLo + 1);
        plan.quartChunk = std::max(DEFAULT_SLICE_QUART_CHUNK, later.rHi - later.rLo + 1);
    }
    return true;
}

bool CubicRootSliceWorker::Impl::planForSlot(int cubicRootId, int slot, SlotPlan& plan) {
    const auto key = std::make_pair(cubicRootId, slot);
    auto it = slotPlans_.find(key);
    if (it != slotPlans_.end()) {
        plan = it->second;
        return true;
    }
    if (!inferPlanFromDb(cubicRootId, slot, plan)) {
        return false;
    }
    slotPlans_[key] = plan;
    return true;
}

std::optional<long long> CubicRootSliceWorker::Impl::insertSliceRow(
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
    if (mysql_query(mysql_, q.c_str())) {
        std::cerr << "Error: slice INSERT failed: " << mysql_error(mysql_) << std::endl;
        return std::nullopt;
    }
    if (mysql_affected_rows(mysql_) == 0) {
        return 0;
    }
    return static_cast<long long>(mysql_insert_id(mysql_));
}

bool CubicRootSliceWorker::Impl::insertSliceBatchUnstarted(int cubicRootId, int slot, const std::vector<TileRect>& tiles) {
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
    if (mysql_query(mysql_, q.c_str())) {
        std::cerr << "Error: slice batch INSERT failed: " << mysql_error(mysql_) << std::endl;
        return false;
    }
    return true;
}

bool CubicRootSliceWorker::Impl::enqueueNextRemainingForRoot(int cubicRootId) {
    bool any = false;
    for (int slot = 1; slot <= 3; ++slot) {
        SlotPlan plan{};
        if (!planForSlot(cubicRootId, slot, plan)) {
            continue;
        }
        TileRect last{};
        if (!loadLastTile(cubicRootId, slot, last)) {
            continue;
        }
        const bool lastIsProbe = sameTile(last, plan.probe);
        std::vector<TileRect> tiles;
        if (!nextRemainingTiles(plan, last, lastIsProbe, SLICE_ENQUEUE_BATCH, tiles)) {
            continue;
        }
        if (!insertSliceBatchUnstarted(cubicRootId, slot, tiles)) {
            return false;
        }
        std::cout << "Enqueued " << tiles.size() << " remaining slice(s) for cubic_root_id " << cubicRootId
                  << " zroot_slot=" << slot << " at " << plan.quintChunk << "x" << plan.quartChunk << std::endl;
        any = true;
    }
    return any;
}

std::vector<CubicRootSliceWorker::Impl::SliceWorkItem> CubicRootSliceWorker::Impl::getSliceWorkToBeDone(std::optional<int> cubicRootId) {
    std::vector<SliceWorkItem> items;

    std::string query = "SELECT id FROM roots_checked_slice WHERE is_started = 0 AND is_finished = 0";
    if (cubicRootId.has_value()) {
        query += " AND cubic_root_id = " + std::to_string(*cubicRootId);
    }
    query += " ORDER BY id ASC LIMIT " + std::to_string(kWorkItemsPerQuery);

    if (mysql_query(mysql_, query.c_str())) {
        std::cerr << "Error: roots_checked_slice query failed: " << mysql_error(mysql_) << std::endl;
        return items;
    }

    MYSQL_RES* result = mysql_store_result(mysql_);
    if (!result) {
        std::cerr << "Error: mysql_store_result failed: " << mysql_error(mysql_) << std::endl;
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

    std::string updateQuery = "UPDATE roots_checked_slice SET is_started = 1, worker_id = " + std::to_string(workerId_) + " WHERE id IN (";
    for (size_t i = 0; i < ids.size(); ++i) {
        updateQuery += std::to_string(ids[i]);
        if (i < ids.size() - 1) {
            updateQuery += ",";
        }
    }
    updateQuery += ")";

    if (mysql_query(mysql_, updateQuery.c_str())) {
        std::cerr << "Error: roots_checked_slice UPDATE failed: " << mysql_error(mysql_) << std::endl;
        return items;
    }

    std::cout << "Updated " << ids.size() << " rows in roots_checked_slice to is_started = 1 and worker_id = " << workerId_ << std::endl;

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

    if (mysql_query(mysql_, selectQuery.c_str())) {
        std::cerr << "Error: slice join query failed: " << mysql_error(mysql_) << std::endl;
        return items;
    }

    MYSQL_RES* selectResult = mysql_store_result(mysql_);
    if (!selectResult) {
        std::cerr << "Error: mysql_store_result failed: " << mysql_error(mysql_) << std::endl;
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

bool CubicRootSliceWorker::Impl::updateSliceFinished(long long sliceId, long doubleHits, long float128Hits) {
    std::string q = "UPDATE roots_checked_slice SET is_finished = 1, double_hit_count = " + std::to_string(doubleHits)
        + ", float128_hit_count = " + std::to_string(float128Hits) + " WHERE id = " + std::to_string(sliceId);
    if (mysql_query(mysql_, q.c_str())) {
        std::cerr << "Error: slice finish UPDATE failed: " << mysql_error(mysql_) << std::endl;
        return false;
    }
    return true;
}

bool CubicRootSliceWorker::Impl::addSliceHitsToRootsChecked(int cubicRootId, int zrootSlot, long doubleHits, long float128Hits) {
    if (zrootSlot < 1 || zrootSlot > 3) {
        std::cerr << "Error: invalid zroot_slot " << zrootSlot << std::endl;
        return false;
    }
    std::string dcol = "double_hit_count" + std::to_string(zrootSlot);
    std::string fcol = "float128_hit_count" + std::to_string(zrootSlot);
    std::string q = "UPDATE roots_checked SET " + dcol + " = COALESCE(" + dcol + ", 0) + " + std::to_string(doubleHits)
        + ", " + fcol + " = COALESCE(" + fcol + ", 0) + " + std::to_string(float128Hits) + " WHERE id = " + std::to_string(cubicRootId);
    if (mysql_query(mysql_, q.c_str())) {
        std::cerr << "Error: roots_checked incremental UPDATE failed: " << mysql_error(mysql_) << std::endl;
        return false;
    }
    return true;
}

bool CubicRootSliceWorker::Impl::finalizeRootsCheckedIfAllSlicesDone(int cubicRootId) {
    std::string cntQuery = "SELECT COUNT(*) FROM roots_checked_slice WHERE cubic_root_id = " + std::to_string(cubicRootId)
        + " AND is_finished = 0";
    if (mysql_query(mysql_, cntQuery.c_str())) {
        std::cerr << "Error: slice count query failed: " << mysql_error(mysql_) << std::endl;
        return false;
    }
    MYSQL_RES* res = mysql_store_result(mysql_);
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
    if (mysql_query(mysql_, fin.c_str())) {
        std::cerr << "Error: roots_checked finalize UPDATE failed: " << mysql_error(mysql_) << std::endl;
        return false;
    }
    std::cout << "All slices currently in the table for cubic_root_id " << cubicRootId
              << " are finished; set roots_checked.is_finished = 1" << std::endl;
    return true;
}

void CubicRootSliceWorker::Impl::freeHitsVector(std::vector<int*>* hits) {
    if (!hits) {
        return;
    }
    for (int* p : *hits) {
        delete[] p;
    }
    delete hits;
}

CubicRootSliceWorker::Impl::SliceSearchResult CubicRootSliceWorker::Impl::executeSliceSearch(const SliceWorkItem& s) {
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
    std::vector<int*>* hits = checker_->findHits(ZETA5, theConst, 5, getLookupTableFloat(), &loopRanges, doubleHitCount);
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

bool CubicRootSliceWorker::Impl::persistSliceResult(const SliceWorkItem& s, const SliceSearchResult& result) {
    if (!updateSliceFinished(s.sliceId, result.doubleHits, result.float128Hits)) {
        return false;
    }
    return addSliceHitsToRootsChecked(s.cubicRootId, s.zrootSlot, result.doubleHits, result.float128Hits);
}

bool CubicRootSliceWorker::Impl::slotHasSlices(int cubicRootId, int slot) {
    return fetchSingleInt(
        "SELECT 1 FROM roots_checked_slice WHERE cubic_root_id = " + std::to_string(cubicRootId)
        + " AND zroot_slot = " + std::to_string(slot) + " LIMIT 1").has_value();
}

bool CubicRootSliceWorker::Impl::probeZrootSlot(int cubicRootId, int slot, double zrootVal) {
    const TileRect probe = makeProbeTile(bounds_);
    const std::optional<long long> sliceId = insertSliceRow(cubicRootId, slot, probe, true, workerId_);
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
    const SliceSearchResult result = executeSliceSearch(item);
    const double probeSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!persistSliceResult(item, result)) {
        return false;
    }

    const auto chunks = chunksFromProbe(probeSec, bounds_, targetSliceRuntimeSec());
    SlotPlan plan;
    plan.quintChunk = chunks.first;
    plan.quartChunk = chunks.second;
    plan.bounds = bounds_;
    plan.probe = probe;
    slotPlans_[std::make_pair(cubicRootId, slot)] = plan;

    std::cout << "Probe for cubic_root_id " << cubicRootId << " zroot_slot=" << slot << " took " << probeSec
              << "s; remaining tiles " << plan.quintChunk << "x" << plan.quartChunk << " (target "
              << targetSliceRuntimeSec() << "s per tile)" << std::endl;
    return true;
}

void CubicRootSliceWorker::Impl::processCubicRoot(int cubicRootId) {
    std::cout << "Processing cubic_root_id " << cubicRootId
              << "; this worker will keep claiming its slices until the root is done." << std::endl;

    const auto zroots = loadZroots(cubicRootId);
    bool anySlot = false;
    for (int slot = 1; slot <= 3; ++slot) {
        if (!zroots[slot].has_value()) {
            continue;
        }
        anySlot = true;
        if (!slotHasSlices(cubicRootId, slot)) {
            if (!probeZrootSlot(cubicRootId, slot, zroots[slot].value())) {
                return;
            }
        }
    }

    if (!anySlot) {
        std::cout << "No zroot values for cubic_root_id " << cubicRootId << "; marking finished." << std::endl;
        finalizeRootsCheckedIfAllSlicesDone(cubicRootId);
        std::string fin = "UPDATE roots_checked SET is_finished = 1 WHERE id = " + std::to_string(cubicRootId);
        if (mysql_query(mysql_, fin.c_str())) {
            std::cerr << "Error: empty-root finalize UPDATE failed: " << mysql_error(mysql_) << std::endl;
        }
        return;
    }

    for (;;) {
        const std::vector<SliceWorkItem> items = getSliceWorkToBeDone(cubicRootId);
        if (!items.empty()) {
            for (const SliceWorkItem& s : items) {
                runOneSlice(s);
            }
            continue;
        }
        if (enqueueNextRemainingForRoot(cubicRootId)) {
            continue;
        }
        break;
    }
    finalizeRootsCheckedIfAllSlicesDone(cubicRootId);
}

void CubicRootSliceWorker::Impl::runOneSlice(const SliceWorkItem& s) {
    persistSliceResult(s, executeSliceSearch(s));
}

CubicRootSliceWorker::CubicRootSliceWorker(MYSQL* mysql, PolynomialCheckerInterface* checker)
    : impl_(std::make_unique<Impl>(mysql, checker))
{
}

CubicRootSliceWorker::~CubicRootSliceWorker() = default;

bool CubicRootSliceWorker::resolveWorkerId() {
    return impl_->resolveWorkerId();
}

bool CubicRootSliceWorker::runOneCubicRoot() {
    return impl_->runOneCubicRoot();
}
