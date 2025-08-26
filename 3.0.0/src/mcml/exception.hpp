#pragma once

#include <stdexcept>

/**
 * @brief Custom exception class with retry capability
 * 
 * Extends std::runtime_error to provide additional context about whether
 * the operation that failed should be retried.
 */
class Exception : public std::runtime_error
{
public:
	Exception() : std::runtime_error({}) {}
	
	/**
	 * @brief Construct exception with message and retry flag
	 * @param message Error message describing the exception
	 * @param retry Whether the failed operation should be retried
	 */
	Exception(const std::string message, bool retry = false) : std::runtime_error(message), m_retry {retry} {}
	
	/** @brief Check if operation should be retried (implicit conversion to bool) */
	operator bool() const { return m_retry; }
	
	/** @brief Check if operation should be retried */
	bool retry() const { return m_retry; }

private:
	bool m_retry {false};
};
