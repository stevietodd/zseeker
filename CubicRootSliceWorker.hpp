#ifndef CUBIC_ROOT_SLICE_WORKER_HPP
#define CUBIC_ROOT_SLICE_WORKER_HPP

#include <memory>
#include <mysql/mysql.h>
#include <string>
#include "PolynomialCheckerInterface.hpp"

// Credentials for reopening the connection after a long slice search.
struct MysqlConnectInfo {
    std::string host;
    std::string user;
    std::string password;
    std::string database;
    unsigned int port = 3306;
};

// Claims one cubic root and drains its zroot slots as 1x1000 slices.
// Takes ownership of mysql and closes it in the destructor. A slice search
// closes the connection first and opens a new one before writing results.
class CubicRootSliceWorker
{
public:
    CubicRootSliceWorker(MYSQL* mysql, MysqlConnectInfo connectInfo, PolynomialCheckerInterface* checker);
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
