#pragma once
// Diagnostic adapter only: DeepJIT's upstream exception.hpp requires Linux
// elfutils for symbolized backtraces. The cache needs only these three macros.
// Keep cache/hash/filesystem code unchanged and avoid a system libdw dependency.
#include <format>
#include <stdexcept>
#include <string>

#define DJ_STATIC_ASSERT(cond, ...) static_assert(cond, "" __VA_ARGS__)
#define DJ_HOST_ASSERT(cond, ...) \
do { \
    if (!(cond)) \
        throw std::runtime_error(std::string(#cond) __VA_OPT__(+ ": " + std::format(__VA_ARGS__))); \
} while (0)
#define DJ_PANIC(...) throw std::runtime_error(std::format(__VA_ARGS__))
