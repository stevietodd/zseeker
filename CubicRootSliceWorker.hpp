#ifndef CUBIC_ROOT_SLICE_WORKER_HPP
#define CUBIC_ROOT_SLICE_WORKER_HPP

#include <memory>
#include <mysql/mysql.h>
#include "PolynomialCheckerInterface.hpp"

// Claims one cubic root, probes each zroot slot with a timed 1x30 tile, then
// drains remaining quint x quart slices for that root.
class CubicRootSliceWorker
{
public:
    CubicRootSliceWorker(MYSQL* mysql, PolynomialCheckerInterface* checker);
    ~CubicRootSliceWorker();

    CubicRootSliceWorker(const CubicRootSliceWorker&) = delete;
    CubicRootSliceWorker& operator=(const CubicRootSliceWorker&) = delete;

    bool resolveWorkerId();
    bool runOneCubicRoot();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif
