#include <iostream>
#include <string>
#include <strings.h>
#include <mysql/mysql.h>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <optional>
#include <unistd.h>
#include "CpuPolynomialChecker.hpp"
#include "GpuPolynomialChecker.hpp"
#include "math.hpp"
#include "lookupTable.hpp"

// Structure to hold work data from the database
struct WorkItem {
    int id;
    std::optional<double> zroot1;
    std::optional<double> zroot2;
    std::optional<double> zroot3;
};

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

// Fetch work items from roots_checked table and corresponding zroot values
std::vector<WorkItem> getWorkToBeDone(MYSQL* mysql) {
    std::vector<WorkItem> workItems;
    
    // Step 1: Query roots_checked for rows where is_started = 0 with limit 10
    const char* query = "SELECT id FROM roots_checked WHERE is_started = 0 ORDER BY id ASC LIMIT 10";
    
    if (mysql_query(mysql, query)) {
        std::cerr << "Error: mysql_query failed: " << mysql_error(mysql) << std::endl;
        return workItems;
    }
    
    MYSQL_RES* result = mysql_store_result(mysql);
    if (!result) {
        std::cerr << "Error: mysql_store_result failed: " << mysql_error(mysql) << std::endl;
        return workItems;
    }
    
    std::vector<int> ids;
    MYSQL_ROW row;
    while ((row = mysql_fetch_row(result))) {
        if (row[0]) {
            ids.push_back(std::stoi(row[0]));
        }
    }
    mysql_free_result(result);
    
    if (ids.empty()) {
        std::cout << "No work items found with is_started = 0" << std::endl;
        return workItems;
    }
    
    // Step 2: Query workers table to get worker_id for this hostname
    // Get hostname using gethostname() system call
    char hostnameBuffer[61]; // 60 chars + null terminator
    if (gethostname(hostnameBuffer, sizeof(hostnameBuffer)) != 0) {
        std::cerr << "Error: gethostname() failed" << std::endl;
        return workItems;
    }
    std::string hostname(hostnameBuffer);
    
    // Escape hostname for SQL (use mysql_real_escape_string for safety)
    char escapedHostname[121]; // 60*2 + 1 for worst case
    unsigned long escapedLen = mysql_real_escape_string(mysql, escapedHostname, hostname.c_str(), hostname.length());
    std::string workerQuery = "SELECT id FROM workers WHERE hostname = '" + std::string(escapedHostname, escapedLen) + "' LIMIT 1";
    int workerId = -1;
    
    if (mysql_query(mysql, workerQuery.c_str())) {
        std::cerr << "Error: workers query failed: " << mysql_error(mysql) << std::endl;
        return workItems;
    }
    
    MYSQL_RES* workerResult = mysql_store_result(mysql);
    if (!workerResult) {
        std::cerr << "Error: mysql_store_result failed for workers query: " << mysql_error(mysql) << std::endl;
        return workItems;
    }
    
    MYSQL_ROW workerRow = mysql_fetch_row(workerResult);
    if (workerRow && workerRow[0]) {
        workerId = std::stoi(workerRow[0]);
        std::cout << "Found worker_id: " << workerId << std::endl;
    } else {
        std::cerr << "Error: No workers found for hostname = " << hostname << std::endl;
        mysql_free_result(workerResult);
        return workItems;
    }
    mysql_free_result(workerResult);
    
    // Step 3: Update is_started = 1 and worker_id for the fetched IDs
    std::string updateQuery = "UPDATE roots_checked SET is_started = 1, worker_id = " + std::to_string(workerId) + " WHERE id IN (";
    for (size_t i = 0; i < ids.size(); ++i) {
        updateQuery += std::to_string(ids[i]);
        if (i < ids.size() - 1) {
            updateQuery += ",";
        }
    }
    updateQuery += ")";
    
    if (mysql_query(mysql, updateQuery.c_str())) {
        std::cerr << "Error: UPDATE failed: " << mysql_error(mysql) << std::endl;
        return workItems;
    }
    
    std::cout << "Updated " << ids.size() << " rows in roots_checked to is_started = 1 and worker_id = " << workerId << std::endl;
    
    // Step 4: Fetch zroot1, zroot2, zroot3 from z3_cubic_roots for all ids in one query
    std::string selectQuery = "SELECT id, zroot1, zroot2, zroot3 FROM z3_cubic_roots WHERE id IN (";
    for (size_t i = 0; i < ids.size(); ++i) {
        selectQuery += std::to_string(ids[i]);
        if (i < ids.size() - 1) {
            selectQuery += ",";
        }
    }
    selectQuery += ") ORDER BY id";
    
    if (mysql_query(mysql, selectQuery.c_str())) {
        std::cerr << "Error: mysql_query failed: " << mysql_error(mysql) << std::endl;
        return workItems;
    }
    
    MYSQL_RES* selectResult = mysql_store_result(mysql);
    if (!selectResult) {
        std::cerr << "Error: mysql_store_result failed: " << mysql_error(mysql) << std::endl;
        return workItems;
    }
    
    MYSQL_ROW selectRow;
    while ((selectRow = mysql_fetch_row(selectResult))) {
        if (selectRow[0]) {
            WorkItem item;
            item.id = std::stoi(selectRow[0]);
            // Handle NULL values - if a field is NULL, the optional will remain unset
            item.zroot1 = (selectRow[1] != nullptr) ? std::optional<double>(std::stod(selectRow[1])) : std::nullopt;
            item.zroot2 = (selectRow[2] != nullptr) ? std::optional<double>(std::stod(selectRow[2])) : std::nullopt;
            item.zroot3 = (selectRow[3] != nullptr) ? std::optional<double>(std::stod(selectRow[3])) : std::nullopt;
            workItems.push_back(item);
        }
    }
    
    mysql_free_result(selectResult);
    
    std::cout << "Fetched " << workItems.size() << " work items from z3_cubic_roots" << std::endl;
    
    return workItems;
}

int main(int argc, char *argv[])
{
    PolynomialCheckerInterface *checker;
    std::vector<int*> *hits;
    long floatHitCount = 0;
	double theConst;

	typedef std::numeric_limits< float > ldbl;
	std::cout.precision(ldbl::max_digits10);

    // Initialize MySQL connection and fetch theConst from database
    MySQLConfig config; // This will read from environment variables and validate them
    MYSQL* mysql = initializeMySQLConnection(config);
    if (!mysql) {
        std::cerr << "Failed to initialize MySQL connection. Exiting." << std::endl;
        return 1;
    }
    
    std::cout << "Connected to MySQL database successfully." << std::endl;
    
    // Fetch work items from database
    std::vector<WorkItem> workItems = getWorkToBeDone(mysql);
    
    if (workItems.empty()) {
        std::cout << "No work items to process. Exiting." << std::endl;
        mysql_close(mysql);
        return 0;
    }

    // Process each work item
    for (size_t itemIdx = 0; itemIdx < workItems.size(); ++itemIdx) {
        const WorkItem& item = workItems[itemIdx];
        std::cout << "\n=== Processing work item " << itemIdx + 1 << " (id: " << item.id << ") ===" << std::endl;
        
        // Arrays to store hit counts for each zroot (3 zroots, 2 counts each = 6 values)
        // Use optional to track which zroots were processed
        std::optional<long> floatHitCounts[3];
        std::optional<long> doubleHitCounts[3];
        
        // Process each zroot value - only if it's not NULL
        std::optional<double> zroots[3] = {item.zroot1, item.zroot2, item.zroot3};
        
        for (int zrootIdx = 0; zrootIdx < 3; ++zrootIdx) {
            if (zroots[zrootIdx].has_value()) {
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
		} else if (strcasecmp(argv[1], "gfwb") == 0) {
			std::cout << "Using GpuQuinticFirstWithBreakouts" << std::endl;
			checker = new GpuQuinticFirstWithBreakoutsChecker();
		} else if (strcasecmp(argv[1], "cl") == 0) {
			std::cout << "Using CpuQuinticLast" << std::endl;
			checker = new CpuQuinticLastChecker();
		} else if (strcasecmp(argv[1], "cf") == 0) {
			std::cout << "Using CpuQuinticFirst" << std::endl;
			checker = new CpuQuinticFirstChecker();
		} else if (strcasecmp(argv[1], "cfwb") == 0) {
			std::cout << "Using CpuQuinticFirstWithBreakouts" << std::endl;
			checker = new CpuQuinticFirstWithBreakoutsChecker();
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

                theConst = zroots[zrootIdx].value();
                std::cout << "\nProcessing zroot" << zrootIdx + 1 << " = " << theConst << std::endl;
                
                floatHitCount = 0;
                hits = checker->findHits(ZETA5, theConst, 5, LUT.data(), NULL, floatHitCount);
                
                // Store the hit counts
                floatHitCounts[zrootIdx] = floatHitCount;
                doubleHitCounts[zrootIdx] = hits->size();
                
                int *result;
                for (int i = 0; i < hits->size(); i++) {
                    result = hits->at(i);
                    std::cout << "Hit = " << result[0] << "," << result[1] << "," << result[2] << "," 
                              << result[3] << "," << result[4] << "," << result[5] << "," << std::endl;
                }

				// clean up
				delete checker;
            } else {
                std::cout << "\nSkipping zroot" << zrootIdx + 1 << " (NULL value)" << std::endl;
            }
        }
        
        // Build UPDATE query - only include columns for non-NULL zroots
        std::string updateQuery = "UPDATE roots_checked SET ";
        std::vector<std::string> updateFields;
        
        if (floatHitCounts[0].has_value()) {
            updateFields.push_back("float_hit_count1 = " + std::to_string(floatHitCounts[0].value()));
        }
        if (floatHitCounts[1].has_value()) {
            updateFields.push_back("float_hit_count2 = " + std::to_string(floatHitCounts[1].value()));
        }
        if (floatHitCounts[2].has_value()) {
            updateFields.push_back("float_hit_count3 = " + std::to_string(floatHitCounts[2].value()));
        }
        if (doubleHitCounts[0].has_value()) {
            updateFields.push_back("double_hit_count1 = " + std::to_string(doubleHitCounts[0].value()));
        }
        if (doubleHitCounts[1].has_value()) {
            updateFields.push_back("double_hit_count2 = " + std::to_string(doubleHitCounts[1].value()));
        }
        if (doubleHitCounts[2].has_value()) {
            updateFields.push_back("double_hit_count3 = " + std::to_string(doubleHitCounts[2].value()));
        }
        
        // Always set is_finished = 1
        updateFields.push_back("is_finished = 1");
        
        // Join all fields with commas
        for (size_t i = 0; i < updateFields.size(); ++i) {
            updateQuery += updateFields[i];
            if (i < updateFields.size() - 1) {
                updateQuery += ", ";
            }
        }
        
        updateQuery += " WHERE id = " + std::to_string(item.id);
        
        if (mysql_query(mysql, updateQuery.c_str())) {
            std::cerr << "Error: UPDATE failed for work item id " << item.id << ": " << mysql_error(mysql) << std::endl;
        } else {
            std::cout << "Updated work item id " << item.id << " with hit counts and set is_finished = 1" << std::endl;
        }
    }
	// result = hits->at(27);
    // std::cout << "Vals=" << result[0] << "," << result[1] << "," << result[2] << "," << result[3] << "," << result[4] << "," << result[5] << "," << std::endl;
    
    // clean up
    checker->~PolynomialCheckerInterface();
    
    // Clean up MySQL connection
    mysql_close(mysql);
    std::cout << "MySQL connection closed." << std::endl;

    return 0;
}