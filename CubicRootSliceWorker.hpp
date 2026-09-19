#ifndef CUBIC_ROOT_SLICE_WORKER_HPP
#define CUBIC_ROOT_SLICE_WORKER_HPP

#include <memory>
#include <mysql/mysql.h>
#include "PolynomialCheckerInterface.hpp"

// Claims one cubic root and drains its zroot slots as 1x100 quint x quart slices.
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
