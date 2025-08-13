#pragma once

#include <stdexcept>

class Exception : public std::runtime_error
{
public:
    Exception() : std::runtime_error({}) { }

    Exception(const std::string message, bool retry = false) 
        : std::runtime_error(message), m_retry{ retry } { }

    operator bool() const
    {
        return m_retry;
    }

    bool retry() const
    {
        return m_retry;
    }

private:
    bool m_retry{ false };
};
